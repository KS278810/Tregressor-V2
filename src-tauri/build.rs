use std::io::Write;
use std::path::Path;

fn main() {
    tauri_build::build();

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let zip_path = Path::new(&out_dir).join("python-embed.zip");
    let src_dir  = Path::new("../dist_portable/T-regressor/python-embed");

    println!("cargo:rerun-if-changed=../dist_portable/T-regressor/python-embed");

    if src_dir.exists() {
        let n = count_files(src_dir);
        println!("cargo:warning=python-embed.zip を作成中 ({n} ファイル) — 数分かかります...");
        create_zip(src_dir, &zip_path);
    } else {
        eprintln!("WARNING: python-embed が見つかりません: {src_dir:?}");
        let f = std::fs::File::create(&zip_path).unwrap();
        zip::ZipWriter::new(f).finish().unwrap();
    }

    let zip_size = std::fs::metadata(&zip_path).map(|m| m.len()).unwrap_or(0);
    println!("cargo:rustc-env=PYTHON_EMBED_ZIP={}", zip_path.display());
    println!("cargo:rustc-env=EMBED_VERSION={zip_size}");
    println!("cargo:warning=python-embed.zip: {} MB", zip_size / 1_000_000);
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

fn zip_dir_recursive<W: Write + std::io::Seek>(
    zip:  &mut zip::ZipWriter<W>,
    base: &Path,
    cur:  &Path,
    opts: &zip::write::SimpleFileOptions,
) {
    let Ok(entries) = std::fs::read_dir(cur) else { return };
    for e in entries.flatten() {
        let path = e.path();
        let rel  = path.strip_prefix(base).unwrap();
        let name = rel.to_str().unwrap().replace('\\', "/");
        if path.is_dir() {
            let _ = zip.add_directory(format!("{name}/"), *opts);
            zip_dir_recursive(zip, base, &path, opts);
        } else if zip.start_file(&name, *opts).is_ok() {
            if let Ok(bytes) = std::fs::read(&path) {
                let _ = zip.write_all(&bytes);
            }
        }
    }
}
