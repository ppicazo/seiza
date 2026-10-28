use std::process::Command;
#[test]
fn combines_standalone_and_cli_reports_without_sources_and_protects_files() {
    let dir = tempfile::tempdir().unwrap();
    let report:serde_json::Value=serde_json::from_str(r#"{"schema_version":1,"stacking_version":"test","output_version":"test","stage":"online","width":2,"height":1,"channels":1,"accepted_frames":2,"rejected_frames":1,"support":[{"supported_pixels":2,"total_pixels":2,"support_minimum":1.0,"support_maximum":2.0,"support_mean":1.5,"rejected_samples":1}],"support_map":{"width":2,"height":1,"channels":1,"values":[2.0,1.0],"quantity":"accepted_observations"},"rejection_map":{"width":2,"height":1,"channels":1,"values":[0.0,1.0],"quantity":"rejected_observations"},"context":null,"final_samples":null,"provenance":null,"warnings":[] }"#).unwrap();
    let a = dir.path().join("s.json");
    let b = dir.path().join("h.json");
    let html = dir.path().join("sho.html");
    let json = dir.path().join("sho.json");
    std::fs::write(&a, report.to_string()).unwrap();
    std::fs::write(&b, serde_json::json!({"post_stack":report}).to_string()).unwrap();
    let run = |label: &str, output: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_seiza"))
            .args([
                "stack-report",
                "--stack",
                &format!("S={}", a.display()),
                "--stack",
                &format!("{label}={}", b.display()),
                "--html",
            ])
            .arg(output)
            .arg("--json")
            .arg(&json)
            .output()
            .unwrap()
    };
    let result = run("H", &html);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let combined: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&json).unwrap()).unwrap();
    assert_eq!(combined["stacks"][0]["label"], "S");
    assert_eq!(combined["stacks"][1]["report"], report);
    let before = std::fs::read(&a).unwrap();
    assert!(!run("H", &a).status.success());
    assert_eq!(std::fs::read(&a).unwrap(), before);
    assert!(!run("s", &dir.path().join("invalid.html")).status.success());
    assert!(!dir.path().join("invalid.html").exists());
}
