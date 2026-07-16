use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::Value;
use tauri::{AppHandle, Emitter, Manager, State};

#[cfg(windows)]
use std::os::windows::process::CommandExt;
#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
#[cfg(windows)]
const DETACHED_PROCESS: u32 = 0x0000_0008;

// ── 埋め込みリソース ──────────────────────────────────────────────────────────

const EMBED_VERSION:    &str  = env!("EMBED_VERSION");
const EMB_PYTHON_ZIP:   &[u8] = include_bytes!(env!("PYTHON_EMBED_ZIP"));
const EMB_INDEX_HTML:   &[u8] = include_bytes!("../../frontend/index.html");
const EMB_LOGO2:        &[u8] = include_bytes!("../../frontend/reference/logo2.png");
const EMB_GIF_OK:       &[u8] = include_bytes!("../../frontend/reference/robo2_ok.gif");
const EMB_GIF_TRAINING: &[u8] = include_bytes!("../../frontend/reference/robo2_training.gif");
const EMB_GIF_DONE:     &[u8] = include_bytes!("../../frontend/reference/robo2_completed.gif");
const EMB_TRAIN_PY:     &[u8] = include_bytes!("../../train_bridge.py");
const EMB_PREDICT_PY:   &[u8] = include_bytes!("../../predict_template.py");
const EMB_LIGHT_PY:     &[u8] = include_bytes!("../../_light.py");
const EMB_BAT_TMPL:     &[u8] = include_bytes!("../../bat_template.txt");
const EMB_NATIVE_EXE:   &[u8] = include_bytes!("../../native_predictor/predict_native.exe");

// ── 状態管理 ──────────────────────────────────────────────────────────────────
// Child と「意図的に止めた」フラグを対で持つ。キャンセルや再実行での kill を
// クラッシュと誤検知して train_error を出さないためのガード。

struct TrainProcess(Mutex<Option<(Child, Arc<AtomicBool>)>>);
struct PredictProcess(Mutex<Option<(Child, Arc<AtomicBool>)>>);

fn take_and_kill(slot: &Mutex<Option<(Child, Arc<AtomicBool>)>>) {
    // Mutexがpoison化(他スレッドがlock中にpanic)していても、子プロセスの後始末は
    // 継続できるべきなので、素直に中身を取り出して回収する。
    if let Some((mut child, cancelled)) = slot.lock().unwrap_or_else(|e| e.into_inner()).take() {
        cancelled.store(true, Ordering::SeqCst);
        kill_process(&mut child);
        wait_in_background(child);
    }
}

// taskkillでプロセスを終了させた後、child.wait()を呼ばないままChildをdropすると
// (Windows実装上は問題ないが)ゾンビ化やハンドルリークの温床になりうるため、
// 別スレッドでwait()して確実にハンドルを解放する。
fn wait_in_background(mut child: Child) {
    thread::spawn(move || { let _ = child.wait(); });
}

// ── パス解決 ─────────────────────────────────────────────────────────────────

fn res_dir() -> PathBuf {
    let base = std::env::var("LOCALAPPDATA")
        .or_else(|_| std::env::var("APPDATA"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(base).join("T-regressor")
}

fn python_exe() -> PathBuf {
    let res = res_dir().join("python-embed");
    // pythonw.exe はコンソールウィンドウを表示しない（GUIサブシステム）
    let pw = res.join("pythonw.exe");
    if pw.exists() { return pw; }
    let p = res.join("python.exe");
    if p.exists() { p } else { PathBuf::from("python") }
}

fn model_dir_path() -> PathBuf { res_dir().join("trained_model") }

// ── 自己展開 ─────────────────────────────────────────────────────────────────

fn version_ok() -> bool {
    std::fs::read_to_string(res_dir().join(".version"))
        .map(|v| v.trim() == EMBED_VERSION)
        .unwrap_or(false)
}

fn extract_scripts(dir: &PathBuf) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("train_bridge.py"),     EMB_TRAIN_PY)?;
    std::fs::write(dir.join("predict_template.py"), EMB_PREDICT_PY)?;
    std::fs::write(dir.join("_light.py"),           EMB_LIGHT_PY)?;
    std::fs::write(dir.join("bat_template.txt"),    EMB_BAT_TMPL)?;
    let nd = dir.join("native_dist");
    std::fs::create_dir_all(&nd)?;
    std::fs::write(nd.join("predict_native.exe"), EMB_NATIVE_EXE)?;
    std::fs::create_dir_all(dir.join("trained_model"))?;
    Ok(())
}

fn start_python_extraction(app: AppHandle) {
    let dir = res_dir();
    thread::spawn(move || {
        // 以前は python-embed に直接展開しており、(a) 旧バージョンにあって新バージョンに
        // 無くなったファイルが削除されずに残る(新旧混在)、(b) 展開中に旧python.exeが
        // まだ動いていて対象ファイルがロックされていると File::create() が失敗し、
        // 中途半端に上書きされた状態が残る、という問題があった(低-M13)。
        // ここでは python-embed.tmp に一旦フル展開し、全ファイルの展開に成功した場合だけ
        // 本番の python-embed と一括で入れ替える(rename)。展開失敗時は python-embed
        // 自体には一切手を付けないため、次回起動時も直前の正常な状態のまま再展開できる。
        let py_dir = dir.join("python-embed");
        let tmp_dir = dir.join("python-embed.tmp");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
            let _ = app.emit("extraction_error", format!("展開用フォルダの作成に失敗しました: {e}"));
            return;
        }

        let cursor = std::io::Cursor::new(EMB_PYTHON_ZIP);
        let mut archive = match zip::ZipArchive::new(cursor) {
            Ok(a) => a,
            Err(e) => {
                let _ = app.emit("extraction_error", e.to_string());
                let _ = std::fs::remove_dir_all(&tmp_dir);
                return;
            }
        };

        let total = archive.len();
        if total == 0 {
            let _ = app.emit("extraction_error", "python-embed が埋め込まれていません");
            let _ = std::fs::remove_dir_all(&tmp_dir);
            return;
        }

        let mut had_error = false;
        for i in 0..total {
            let mut f = match archive.by_index(i) {
                Ok(f) => f,
                Err(_) => { had_error = true; continue; }
            };
            let out_path = tmp_dir.join(f.name());
            if f.is_dir() {
                if std::fs::create_dir_all(&out_path).is_err() { had_error = true; }
            } else {
                if let Some(p) = out_path.parent() { let _ = std::fs::create_dir_all(p); }
                match std::fs::File::create(&out_path) {
                    Ok(mut out) => {
                        if std::io::copy(&mut f, &mut out).is_err() { had_error = true; }
                    }
                    Err(_) => { had_error = true; }
                }
            }
            if i % 300 == 0 || i == total - 1 {
                let pct = ((i + 1) * 100) / total;
                let _ = app.emit("extraction_progress", serde_json::json!({
                    "percent": pct,
                    "current": i + 1,
                    "total":   total
                }));
            }
        }

        // 展開に失敗したファイルがある場合は tmp を破棄し、本番の python-embed は
        // 一切変更せず終了する（次回起動時に tmp から再展開させて自己修復する）。
        if had_error {
            let _ = std::fs::remove_dir_all(&tmp_dir);
            let _ = app.emit("extraction_error",
                "一部ファイルの展開に失敗しました。ディスク容量を確認して再起動してください。");
            return;
        }

        // tmp → 本番への入れ替え。旧フォルダが存在する場合は一旦退避してから削除する
        // ことで、rename失敗時に「両方存在しない」状態にならないようにする。
        if py_dir.exists() {
            let old_dir = dir.join("python-embed.old");
            let _ = std::fs::remove_dir_all(&old_dir);
            if let Err(e) = std::fs::rename(&py_dir, &old_dir) {
                let _ = app.emit("extraction_error",
                    format!("旧python-embedの退避に失敗しました: {e}"));
                let _ = std::fs::remove_dir_all(&tmp_dir);
                return;
            }
            if let Err(e) = std::fs::rename(&tmp_dir, &py_dir) {
                // 入れ替え失敗時は旧フォルダを復元してロールバックする
                let _ = std::fs::rename(&old_dir, &py_dir);
                let _ = app.emit("extraction_error", format!("python-embedの入れ替えに失敗しました: {e}"));
                return;
            }
            let _ = std::fs::remove_dir_all(&old_dir);
        } else if let Err(e) = std::fs::rename(&tmp_dir, &py_dir) {
            let _ = app.emit("extraction_error", format!("python-embedの配置に失敗しました: {e}"));
            return;
        }

        let _ = std::fs::write(res_dir().join(".version"), EMBED_VERSION);
        let _ = app.emit("extraction_complete", ());
    });
}

// ── treg:// URI スキームハンドラ ──────────────────────────────────────────────

// tauri.conf.json の app.security.csp と同一内容を保つこと。カスタムプロトコル
// (treg://) 経由の応答には tauri.conf.json の csp 設定が自動付与されないため、
// ここで明示的にヘッダを付与しない限りCSPが実質無効になっていた(M-2対応)。
// index.html は @font-face で data: URI のフォントを埋め込んでいるため
// font-src 'self' data: を含める。
const TREG_CSP: &str = "default-src 'self' ipc: http://ipc.localhost; \
img-src 'self' data:; style-src 'self' 'unsafe-inline'; \
script-src 'self' 'unsafe-inline'; font-src 'self' data:; \
connect-src 'self' ipc: http://ipc.localhost; object-src 'none'; \
base-uri 'self'; form-action 'self'";

fn serve_treg(path: &str) -> tauri::http::Response<Vec<u8>> {
    let (data, mime): (&[u8], &str) = match path {
        "/" | "/index.html" | "" => (EMB_INDEX_HTML, "text/html; charset=utf-8"),
        "/reference/logo2.png"           => (EMB_LOGO2,        "image/png"),
        "/reference/robo2_ok.gif"        => (EMB_GIF_OK,       "image/gif"),
        "/reference/robo2_training.gif"  => (EMB_GIF_TRAINING, "image/gif"),
        "/reference/robo2_completed.gif" => (EMB_GIF_DONE,     "image/gif"),
        _ => return tauri::http::Response::builder()
                .status(404)
                .header("Content-Security-Policy", TREG_CSP)
                .body(b"Not Found".to_vec())
                .unwrap(),
    };
    tauri::http::Response::builder()
        .header("Content-Type", mime)
        .header("Content-Security-Policy", TREG_CSP)
        .status(200)
        .body(data.to_vec())
        .unwrap()
}

// ── プロセス強制終了 ─────────────────────────────────────────────────────────

fn kill_process(child: &mut Child) {
    #[cfg(windows)]
    {
        let mut cmd = Command::new("taskkill");
        cmd.args(["/pid", &child.id().to_string(), "/f", "/t"]);
        cmd.creation_flags(CREATE_NO_WINDOW);
        let _ = cmd.spawn();
    }
    let _ = child.kill();
}

// ── コマンド ─────────────────────────────────────────────────────────────────

#[tauri::command]
async fn check_python_ready() -> bool {
    // python.exe の存在だけでは、旧版展開済み環境にバージョン更新配布した際に
    // バックグラウンド再展開中の状態を「準備完了」と誤認してしまう(中-12)。
    // 展開完了は version_ok() (.version の内容一致) でのみ判定する。
    version_ok()
}

#[tauri::command]
async fn run_train(
    app: AppHandle,
    train_proc: State<'_, TrainProcess>,
    csv_path: String,
    target_col: String,
    strategy: String,
    num_jobs: i32,
) -> Result<(), String> {
    if !csv_path.to_lowercase().ends_with(".csv") {
        return Err("無効なファイルパスです".to_string());
    }

    let num_jobs = num_jobs.clamp(1, 16);
    let python = python_exe();
    let script = res_dir().join("train_bridge.py");
    let mut cmd = Command::new(&python);
    cmd.args([
        script.to_str().unwrap_or("train_bridge.py"),
        &csv_path,
        &target_col,
        "0",
        &strategy,
        &num_jobs.to_string(),
    ])
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .env("PYTHONUTF8", "1").env("PYTHONIOENCODING", "utf-8");
    #[cfg(windows)] cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);

    // kill→spawn→store を単一ロック区間に（中-14）。分離していると、ほぼ同時の
    // 二重呼び出しで先発の子プロセスがどこにも記録されないまま孤児化しうる。
    let cancelled = Arc::new(AtomicBool::new(false));
    let (stdout, stderr) = {
        let mut slot = train_proc.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((mut old_child, old_cancelled)) = slot.take() {
            old_cancelled.store(true, Ordering::SeqCst);
            kill_process(&mut old_child);
            wait_in_background(old_child);
        }
        let mut child = cmd.spawn().map_err(|e| format!("Python起動失敗: {e}"))?;
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        *slot = Some((child, cancelled.clone()));
        (stdout, stderr)
    };

    // stderr: ログ転送しつつ末尾を保持（クラッシュ時のエラーメッセージに使う）
    let err_tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let app3 = app.clone();
    let tail_w = err_tail.clone();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().flatten() {
            if !line.is_empty() {
                let _ = app3.emit("log_data", format!("[err] {line}"));
                let mut t = tail_w.lock().unwrap_or_else(|e| e.into_inner());
                t.push(line);
                if t.len() > 12 { t.remove(0); }
            }
        }
    });

    let app2 = app.clone();
    thread::spawn(move || {
        let mut finished = false;
        for line in BufReader::new(stdout).lines().flatten() {
            if let Some(j) = line.strip_prefix("RESULT_JSON:") {
                match serde_json::from_str::<Value>(j) {
                    Ok(val) => {
                        let _ = app2.emit("train_complete", &val);
                    }
                    Err(e) => { let _ = app2.emit("train_error", format!("JSON解析失敗: {e}")); }
                }
                finished = true;
            } else if line.starts_with("ERROR:") {
                let _ = app2.emit("train_error", &line);
                finished = true;
            } else if !line.is_empty() {
                let _ = app2.emit("log_data", &line);
            }
        }
        // stdout が閉じた = プロセス終了。結果もエラーも出ていなければクラッシュ。
        // これがないと UI は「学習中...」のまま永遠に止まる。
        if !finished && !cancelled.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(700)); // stderr スレッドの取りこぼし防止
            let tail = err_tail.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
            let msg = if tail.is_empty() {
                "学習プロセスが結果を返さずに終了しました。CSV の内容を確認してください。".to_string()
            } else {
                format!("学習プロセスが異常終了しました:\n{tail}")
            };
            let _ = app2.emit("train_error", msg);
        }
    });
    Ok(())
}

#[tauri::command]
async fn cancel_train(train_proc: State<'_, TrainProcess>) -> Result<(), String> {
    take_and_kill(&train_proc.0);
    Ok(())
}

#[tauri::command]
async fn run_predict(
    app: AppHandle,
    pred_proc: State<'_, PredictProcess>,
    csv_path: String,
) -> Result<(), String> {
    if !csv_path.to_lowercase().ends_with(".csv") {
        return Err("無効なファイルパスです".to_string());
    }

    let python = python_exe();
    let script = res_dir().join("predict_template.py");
    let mut cmd = Command::new(&python);
    cmd.args([script.to_str().unwrap_or("predict_template.py"), &csv_path])
        .stdout(Stdio::piped()).stderr(Stdio::piped())
        .env("PYTHONUTF8", "1").env("PYTHONIOENCODING", "utf-8");
    #[cfg(windows)] cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);

    // kill→spawn→store を単一ロック区間に（中-14、run_trainと同様）。
    let cancelled = Arc::new(AtomicBool::new(false));
    let (stdout, stderr) = {
        let mut slot = pred_proc.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((mut old_child, old_cancelled)) = slot.take() {
            old_cancelled.store(true, Ordering::SeqCst);
            kill_process(&mut old_child);
            wait_in_background(old_child);
        }
        let mut child = cmd.spawn().map_err(|e| format!("Python起動失敗: {e}"))?;
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();
        *slot = Some((child, cancelled.clone()));
        (stdout, stderr)
    };

    let err_tail: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let app3 = app.clone();
    let tail_w = err_tail.clone();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines().flatten() {
            if !line.is_empty() {
                let _ = app3.emit("log_data", format!("[err] {line}"));
                let mut t = tail_w.lock().unwrap_or_else(|e| e.into_inner());
                t.push(line);
                if t.len() > 12 { t.remove(0); }
            }
        }
    });

    let app2 = app.clone();
    thread::spawn(move || {
        let mut finished = false;
        for line in BufReader::new(stdout).lines().flatten() {
            if let Some(j) = line.strip_prefix("PREDICT_JSON:") {
                match serde_json::from_str::<Value>(j) {
                    Ok(val) => { let _ = app2.emit("predict_complete", &val); }
                    Err(e) => { let _ = app2.emit("predict_error", format!("JSON解析失敗: {e}")); }
                }
                finished = true;
            } else if line.contains("予測エラー:") {
                // predict_template.py が catch した例外は stdout に
                // "[Robot] 予測エラー: ..." として出るが、従来はlog_dataに流すだけで
                // フロントが無視していた(M-1対応)。run_trainのERROR:処理と同様に
                // predict_errorとして伝搬させ、フロントの汎用「結果を返さず終了」
                // メッセージに化けないようにする。
                let _ = app2.emit("predict_error", &line);
                finished = true;
            } else if !line.is_empty() {
                let _ = app2.emit("log_data", &line);
            }
        }
        if !finished && !cancelled.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(700));
            let tail = err_tail.lock().unwrap_or_else(|e| e.into_inner()).join("\n");
            let msg = if tail.is_empty() {
                "予測プロセスが結果を返さずに終了しました。".to_string()
            } else {
                format!("予測プロセスが異常終了しました:\n{tail}")
            };
            let _ = app2.emit("predict_error", msg);
        }
    });
    Ok(())
}

#[tauri::command]
async fn cancel_predict(pred_proc: State<'_, PredictProcess>) -> Result<(), String> {
    take_and_kill(&pred_proc.0);
    Ok(())
}

#[tauri::command]
async fn read_csv_headers(path: String) -> Result<Vec<String>, String> {
    // ヘッダ抽出のためにファイル全量を読む必要はない。BufReaderで先頭行のみを
    // 読み取ることで、数GB級CSVでもメモリスパイクを起こさない(High-5対応)。
    let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(file);
    let mut raw: Vec<u8> = Vec::new();
    reader.read_until(b'\n', &mut raw).map_err(|e| e.to_string())?;
    if raw.last() == Some(&b'\n') { raw.pop(); }
    if raw.last() == Some(&b'\r') { raw.pop(); }
    let body: &[u8] = if raw.starts_with(&[0xEF, 0xBB, 0xBF]) { &raw[3..] } else { &raw[..] };

    // まずUTF-8として妥当かを検査し、無効なら日本語Excel既定のcp932(Shift-JIS)として
    // デコードする。frontend/index.html の fileText() 内と同じ判定方針
    // (train_bridge.py._read_csv_with_encoding_fallbackとも整合)。
    // encoding_rs::SHIFT_JIS はWHATWG "shift_jis"ラベル(実質cp932相当)としてデコードする。
    let first = match String::from_utf8(body.to_vec()) {
        Ok(s) => s,
        Err(_) => {
            let (decoded, _enc, _had_errors) = encoding_rs::SHIFT_JIS.decode(body);
            decoded.into_owned()
        }
    };

    // クォート対応の最小CSVパース（"a,b" のようなカンマ入り列名を分断しない）
    let mut headers: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = first.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"'); // "" → エスケープされた引用符
                } else {
                    in_quotes = false;
                }
            }
            '"' => in_quotes = true,
            ',' if !in_quotes => {
                headers.push(field.trim().to_string());
                field.clear();
            }
            _ => field.push(c),
        }
    }
    headers.push(field.trim().to_string());
    headers.retain(|h| !h.is_empty());
    Ok(headers)
}

const SAMPLE_CSV: &str = "area_m2,age_years,walk_min,floor,station_rank,rent_10kyen\r\n\
32,3,5,4,4,8.5\r\n45,10,8,2,3,7.2\r\n28,1,3,8,5,9.8\r\n60,15,12,1,3,8.9\r\n\
38,7,6,5,4,8.1\r\n52,2,4,10,5,12.5\r\n25,20,15,2,2,5.8\r\n70,8,7,3,4,11.2\r\n\
41,12,9,6,3,7.8\r\n33,5,4,4,4,8.3\r\n55,0,2,15,5,14.8\r\n48,18,20,1,2,6.5\r\n\
36,4,6,3,4,8.0\r\n65,6,5,7,5,13.2\r\n29,14,11,2,3,6.2\r\n72,3,3,12,5,16.5\r\n\
44,9,10,4,3,7.5\r\n58,1,5,8,4,11.8\r\n31,22,18,1,2,5.2\r\n67,7,6,9,4,12.8\r\n\
40,5,7,5,4,8.6\r\n53,11,9,3,3,8.9\r\n35,3,4,6,4,8.8\r\n78,4,5,14,5,17.2\r\n\
27,16,13,2,2,5.5\r\n62,2,3,11,5,13.8\r\n43,8,8,4,3,7.9\r\n50,13,11,2,3,7.3\r\n\
38,6,5,7,4,9.2\r\n56,0,4,9,5,12.2\r\n";

#[tauri::command]
async fn save_sample_csv() -> Result<Option<String>, String> {
    let dest_file = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .set_title("サンプルCSVの保存先を選択")
            .add_filter("CSV", &["csv"])
            .set_file_name("sample.csv")
            .save_file()
    }).await.map_err(|e| e.to_string())?;

    let dest_file = match dest_file { Some(p) => p, None => return Ok(None) };
    std::fs::write(&dest_file, SAMPLE_CSV).map_err(|e| e.to_string())?;
    Ok(Some(dest_file.to_string_lossy().to_string()))
}

#[tauri::command]
async fn open_csv_dialog() -> Result<Option<String>, String> {
    let result = tauri::async_runtime::spawn_blocking(|| {
        rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .set_title("CSVファイルを選択")
            .pick_file()
    }).await.map_err(|e| e.to_string())?;
    Ok(result.map(|p| p.to_string_lossy().to_string()))
}

// ── デプロイ（EXE 埋め込み） ─────────────────────────────────────────────────

const EXE_TAIL_MAGIC: &[u8] = b"TREG_EMB";

fn embed_treg_into_exe(base_exe: &[u8], treg: &[u8]) -> Vec<u8> {
    let mut out = base_exe.to_vec();
    out.extend_from_slice(treg);
    out.extend_from_slice(&(treg.len() as u64).to_le_bytes());
    out.extend_from_slice(EXE_TAIL_MAGIC);
    out
}

#[tauri::command]
async fn export_robot(_app: AppHandle, file_stem: String) -> Result<Option<String>, String> {
    let trained    = model_dir_path();
    let treg_path  = trained.join("model.treg");
    let native_exe = res_dir().join("native_dist").join("predict_native.exe");

    // 「model.treg が無い」と「native_exeが無い」は原因が違う(前者は未学習、後者は
    // 展開/ビルドの問題)ため、誤誘導しないよう別メッセージにする(低-M15)。
    if !treg_path.exists() {
        return Err("model.treg が見つかりません。再学習してください。".to_string());
    }
    if !native_exe.exists() {
        return Err("予測用の実行ファイル(predict_native.exe)が見つかりません。\
                     インストールが壊れている可能性があります。再インストールしてください。".to_string());
    }

    // model.treg は1回だけ読み、型検査(head)とexeへの埋め込み(treg_bytes)の両方に使う。
    // 以前は型検査時と埋め込み時の2回に分けてファイルを読んでおり、その間(ユーザーが
    // 保存先ダイアログを操作している間を含む長い区間)に別の学習が完了してmodel.tregが
    // 別モデルに置き換わると、型検査をすり抜けたexe書き出しが起こり得た(TOCTOU、低-M15)。
    let treg_bytes = std::fs::read(&treg_path).map_err(|e| e.to_string())?;

    // predict_native.exe (C++) は .treg のモデル型バイト(ヘッダ6バイト目、'TREG'+version+type)
    // として type0(linear)〜type5(blend、type4=linear_poly含む)まで実装している
    // (predict_native_v2.cpp ModelType 参照)。それ以外(将来のフォーマット拡張等)は
    // 動かないexeを「正常生成」してしまう事故を防ぐため、埋め込み前に型を検査する。
    if treg_bytes.len() < 6 || &treg_bytes[0..4] != b"TREG" {
        return Err("model.treg の形式が不正です。再学習してください。".to_string());
    }
    let model_type = treg_bytes[5];
    if model_type > 5 {
        return Err(
            "このモデル種別はexe書き出しに未対応です。\
             HTML版の書き出しをご利用ください。".to_string()
        );
    }

    let stem = if file_stem.is_empty() { "model".to_string() } else { file_stem };
    let default_name = format!("{}.exe", stem);

    let dest_file = tauri::async_runtime::spawn_blocking(move || {
        rfd::FileDialog::new()
            .set_title("予測EXEの保存先を選択")
            .add_filter("実行ファイル", &["exe"])
            .set_file_name(&default_name)
            .save_file()
    }).await.map_err(|e| e.to_string())?;

    let dest_file = match dest_file { Some(p) => p, None => return Ok(None) };
    let exe_bytes  = std::fs::read(&native_exe).map_err(|e| e.to_string())?;
    std::fs::write(&dest_file, embed_treg_into_exe(&exe_bytes, &treg_bytes))
        .map_err(|e| e.to_string())?;
    Ok(Some(dest_file.to_string_lossy().to_string()))
}

// ── エントリーポイント ────────────────────────────────────────────────────────

pub fn run() {
    tauri::Builder::default()
        // 多重起動ガード。2つ目以降の起動は即終了させ、既存インスタンスの
        // メインウィンドウへフォーカスを移譲する。Windows向けの仕様上、
        // 他のプラグインより前に登録する必要がある。
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .manage(TrainProcess(Mutex::new(None)))
        .manage(PredictProcess(Mutex::new(None)))
        .register_uri_scheme_protocol("treg", |_ctx, request| {
            serve_treg(request.uri().path())
        })
        // ウィンドウを閉じる際に学習/予測の子プロセス(pythonw、DETACHED_PROCESS)を
        // 確実にkillする。これがないと孤児化したpythonwが%LOCALAPPDATA%\T-regressor\
        // trained_model に書き続け、再起動後のexport_robotが新旧混在モデルを配布exeに
        // 焼いてしまう事故につながる(High-6対応)。
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let app = window.app_handle();
                take_and_kill(&app.state::<TrainProcess>().0);
                take_and_kill(&app.state::<PredictProcess>().0);
            }
        })
        .invoke_handler(tauri::generate_handler![
            check_python_ready,
            run_train,
            cancel_train,
            run_predict,
            cancel_predict,
            read_csv_headers,
            save_sample_csv,
            export_robot,
            open_csv_dialog,
        ])
        .setup(|app| {
            let dir = res_dir();
            let need_extract = !version_ok();

            // スクリプト類は毎回更新（高速 ~14MB）
            if let Err(e) = extract_scripts(&dir) {
                eprintln!("スクリプト展開失敗: {e}");
            }

            // python-embed はバージョン不一致時のみバックグラウンド展開
            if need_extract {
                let app_handle = app.handle().clone();
                thread::spawn(move || start_python_extraction(app_handle));
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Tauri アプリの起動に失敗")
}
