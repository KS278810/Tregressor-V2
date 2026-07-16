use std::io::Write;
use std::path::Path;

use sha2::{Digest, Sha256};

fn main() {
    tauri_build::build();

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let zip_path = Path::new(&out_dir).join("python-embed.zip");
    let src_dir  = Path::new("../dist_portable/T-regressor/python-embed");

    println!("cargo:rerun-if-changed=../dist_portable/T-regressor/python-embed");
    println!("cargo:rerun-if-env-changed=TREG_ALLOW_EMPTY_EMBED");

    if src_dir.exists() {
        let n = count_files(src_dir);
        println!("cargo:warning=python-embed.zip を作成中 ({n} ファイル) — 数分かかります...");
        create_zip(src_dir, &zip_path);
    } else {
        // python-embedが無いまま「空zipを作って正常終了」していると、実配布ビルドの
        // つもりが埋め込みPython無しのexeを黙って作ってしまう事故になりうる(M-6対応)。
        // 既定はpanicでビルド自体を失敗させ、コンパイルチェックのみ行いたいCI等では
        // 明示的にTREG_ALLOW_EMPTY_EMBED=1を要求する。
        let allow_empty = std::env::var("TREG_ALLOW_EMPTY_EMBED").as_deref() == Ok("1");
        if !allow_empty {
            panic!(
                "python-embed が見つかりません: {src_dir:?}\n\
                 実配布用ビルドには dist_portable/T-regressor/python-embed のセットアップが必要です。\n\
                 コンパイルチェックのみを行う場合は環境変数 TREG_ALLOW_EMPTY_EMBED=1 を設定してください(CI用)。"
            );
        }
        eprintln!(
            "WARNING: python-embed が見つかりません: {src_dir:?} \
             (TREG_ALLOW_EMPTY_EMBED=1 のため空zipで続行します。配布用ビルドには使えません)"
        );
        let f = std::fs::File::create(&zip_path)
            .unwrap_or_else(|e| panic!("空zipファイルの作成に失敗: {e}"));
        zip::ZipWriter::new(f)
            .finish()
            .unwrap_or_else(|e| panic!("空zipのfinishに失敗: {e}"));
    }

    let zip_bytes = std::fs::read(&zip_path)
        .unwrap_or_else(|e| panic!("python-embed.zip の読み取りに失敗: {zip_path:?}: {e}"));
    let zip_size = zip_bytes.len();

    // EMBED_VERSIONはzipバイトサイズではなくSHA-256(先頭16hex)にする。
    // バイトサイズだと「同サイズの別内容」への更新で再展開がスキップされてしまう
    // (M-6対応)。lib.rs::version_ok()の.versionファイル比較と対応させること。
    let mut hasher = Sha256::new();
    hasher.update(&zip_bytes);
    let digest = hasher.finalize();
    let embed_version: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();

    println!("cargo:rustc-env=PYTHON_EMBED_ZIP={}", zip_path.display());
    println!("cargo:rustc-env=EMBED_VERSION={embed_version}");
    println!("cargo:warning=python-embed.zip: {} MB (sha256:{embed_version})", zip_size / 1_000_000);
}

fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| if e.path().is_dir() { count_files(&e.path()) } else { 1 })
        .sum()
}

fn create_zip(src: &Path, dest: &Path) {
    use zip::{CompressionMethod, write::SimpleFileOptions};

    let f   = std::fs::File::create(dest).expect("ZIP 作成失敗");
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(f));
    let opts = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .compression_level(Some(6));

    zip_dir_recursive(&mut zip, src, src, &opts);
    zip.finish().expect("ZIP finish 失敗");
}

// 以前はディレクトリ読み取り失敗・zipへの追加失敗・ファイル読み取り失敗をすべて
// 無言で無視しており(`let _ = ...` / `if ... .is_ok()`)、python-embed.zipが一部
// ファイル欠損のまま「正常にビルド成功」してしまい、その欠損入りzipがexeへ焼かれる
// (エンドユーザーの実機で初めて発覚する)おそれがあった(低-M14)。build.rsはビルド時
// にしか動かないスクリプトなので、ここでのエラーはすべてpanic!でビルド自体を
// 失敗させ、CI/開発者に即座に気づかせる方が安全。
fn zip_dir_recursive<W: Write + std::io::Seek>(
    zip:  &mut zip::ZipWriter<W>,
    base: &Path,
    cur:  &Path,
    opts: &zip::write::SimpleFileOptions,
) {
    let entries = std::fs::read_dir(cur)
        .unwrap_or_else(|e| panic!("python-embed のディレクトリ読み取りに失敗: {cur:?}: {e}"));
    for entry in entries {
        let e = entry.unwrap_or_else(|e| panic!("ディレクトリエントリの読み取りに失敗: {e}"));
        let path = e.path();
        let rel  = path.strip_prefix(base)
            .unwrap_or_else(|e| panic!("パスの相対化に失敗: {path:?}: {e}"));
        let name = rel.to_str()
            .unwrap_or_else(|| panic!("パスがUTF-8ではありません: {rel:?}"))
            .replace('\\', "/");
        if path.is_dir() {
            zip.add_directory(format!("{name}/"), *opts)
                .unwrap_or_else(|e| panic!("zipへのディレクトリ追加に失敗: {name}: {e}"));
            zip_dir_recursive(zip, base, &path, opts);
        } else {
            zip.start_file(&name, *opts)
                .unwrap_or_else(|e| panic!("zipへのファイル追加に失敗: {name}: {e}"));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("ファイル読み取りに失敗: {path:?}: {e}"));
            zip.write_all(&bytes)
                .unwrap_or_else(|e| panic!("zipへの書き込みに失敗: {name}: {e}"));
        }
    }
}
