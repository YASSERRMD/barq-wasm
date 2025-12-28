use crate::patterns::DatabaseDetectionResult;

pub fn detect_mongodb_pattern(bytecode: &[u8]) -> DatabaseDetectionResult {
    let sequence = extract_syscall_sequence(bytecode);

    // Hypothetical sequence for MongoDB: socket (index 1), write (index 2), read (index 3)
    let has_socket = sequence.contains(&"socket".to_string());
    let has_write = sequence.contains(&"write".to_string());
    let has_read = sequence.contains(&"read".to_string());

    let confidence = if has_socket && has_write && has_read {
        0.85
    } else {
        0.1
    };

    DatabaseDetectionResult {
        pattern: "MongoDB Client".to_string(),
        confidence,
    }
}

pub fn detect_filenet_pattern(bytecode: &[u8]) -> DatabaseDetectionResult {
    let sequence = extract_syscall_sequence(bytecode);

    let has_open = sequence.contains(&"open".to_string());
    let has_seek = sequence.contains(&"seek".to_string());
    let has_close = sequence.contains(&"close".to_string());

    let confidence = if has_open && has_seek && has_close {
        0.8
    } else {
        0.1
    };

    DatabaseDetectionResult {
        pattern: "FileNet Storage".to_string(),
        confidence,
    }
}

pub fn detect_file_io_pattern(bytecode: &[u8]) -> DatabaseDetectionResult {
    let sequence = extract_syscall_sequence(bytecode);
    let count = sequence
        .iter()
        .filter(|s| ["open", "read", "write", "close"].contains(&s.as_str()))
        .count();

    let confidence = (count as f32 / 4.0).min(1.0);

    DatabaseDetectionResult {
        pattern: "File I/O".to_string(),
        confidence,
    }
}

pub fn detect_network_io_pattern(bytecode: &[u8]) -> DatabaseDetectionResult {
    let sequence = extract_syscall_sequence(bytecode);
    let count = sequence
        .iter()
        .filter(|s| ["socket", "connect", "send", "recv"].contains(&s.as_str()))
        .count();

    let confidence = (count as f32 / 4.0).min(1.0);

    DatabaseDetectionResult {
        pattern: "Network I/O".to_string(),
        confidence,
    }
}

fn extract_syscall_sequence(bytecode: &[u8]) -> Vec<String> {
    let mut sequence = Vec::new();
    let mut i = 0;
    while i < bytecode.len() {
        if bytecode[i] == 0x10 {
            // call
            // In a real implementation, we'd parse the LEB128 index and look up the import.
            // For this pattern detector, we'll "pseudo-decode" based on the next byte
            // simulating different syscalls.
            if i + 1 < bytecode.len() {
                let pseudo_idx = bytecode[i + 1];
                let name = match pseudo_idx {
                    1 => "socket",
                    2 => "write",
                    3 => "read",
                    4 => "open",
                    5 => "seek",
                    6 => "close",
                    7 => "connect",
                    8 => "send",
                    9 => "recv",
                    _ => "unknown",
                };
                if name != "unknown" {
                    sequence.push(name.to_string());
                }
                i += 1;
            }
        }
        i += 1;
    }
    sequence
}
