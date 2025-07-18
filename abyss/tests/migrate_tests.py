#!/usr/bin/env python3
"""
Script to help migrate test files to the new structure.

This script:
1. Removes sys.path.insert() statements
2. Updates imports to use direct imports
3. Identifies fixtures that can use shared ones from conftest.py
"""

import re
import os
from pathlib import Path
from typing import List, Tuple


def remove_sys_path_inserts(content: str) -> str:
    """Remove sys.path.insert statements and related imports."""
    # Pattern to match sys.path.insert lines and the sys/os imports if only used for path manipulation
    lines = content.split('\n')
    new_lines = []
    skip_next = False
    
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
            
        # Skip sys.path.insert lines
        if 'sys.path.insert' in line:
            # Check if previous line is "import sys" and next line doesn't use sys
            if i > 0 and lines[i-1].strip() == 'import sys':
                # Check if sys is used elsewhere
                sys_used_elsewhere = any('sys.' in l and 'sys.path' not in l 
                                       for j, l in enumerate(lines) if j != i)
                if not sys_used_elsewhere:
                    new_lines.pop()  # Remove the import sys line
            continue
        
        # Skip standalone os imports if only used for path joining
        if line.strip() == 'import os':
            os_used_elsewhere = any('os.' in l and 'os.path.join' not in l and 'os.path.dirname' not in l
                                  for j, l in enumerate(lines) if j > i)
            if not os_used_elsewhere:
                continue
        
        new_lines.append(line)
    
    return '\n'.join(new_lines)


def identify_replaceable_fixtures(content: str) -> List[Tuple[str, str]]:
    """Identify fixtures that can be replaced with shared ones from conftest.py."""
    replaceable = []
    
    # Common fixture patterns
    patterns = [
        (r'@pytest\.fixture\s*\n\s*def sample_config.*?return\s*{[^}]+mqtt[^}]+}', 
         'sample_mqtt_config'),
        (r'@pytest\.fixture\s*\n\s*def.*?config_file.*?NamedTemporaryFile.*?yaml\.dump',
         'temp_config_file'),
        (r'@pytest\.fixture\s*\n\s*def.*?mock_mqtt_client.*?Mock\(\)',
         'mock_mqtt_client'),
    ]
    
    for pattern, replacement in patterns:
        if re.search(pattern, content, re.DOTALL):
            replaceable.append((pattern, replacement))
    
    return replaceable


def migrate_test_file(file_path: Path) -> None:
    """Migrate a single test file to the new structure."""
    print(f"Migrating {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove sys.path.insert statements
    content = remove_sys_path_inserts(content)
    
    # Identify replaceable fixtures
    replaceable = identify_replaceable_fixtures(content)
    
    if replaceable:
        print(f"  Found {len(replaceable)} fixtures that can use shared versions:")
        for pattern, replacement in replaceable:
            print(f"    - Can use '{replacement}' from conftest.py")
    
    # Create backup
    backup_path = file_path.with_suffix('.py.bak')
    with open(backup_path, 'w') as f:
        f.write(content)
    
    print(f"  Created backup at {backup_path}")
    print(f"  Please manually review and update fixtures to use shared ones from conftest.py")
    print()


def find_test_files_to_migrate() -> List[Path]:
    """Find all test files that need migration."""
    test_files = []
    
    # Look for test files with sys.path.insert
    for test_file in Path('abyss/tests').rglob('test_*.py'):
        if test_file.name == 'test_config_manager_clean.py':
            continue  # Skip our example file
            
        with open(test_file, 'r') as f:
            content = f.read()
            if 'sys.path.insert' in content:
                test_files.append(test_file)
    
    return test_files


def main():
    """Main migration function."""
    print("Test Migration Helper")
    print("=" * 50)
    
    # Find files to migrate
    files_to_migrate = find_test_files_to_migrate()
    
    if not files_to_migrate:
        print("No test files found that need migration!")
        return
    
    print(f"Found {len(files_to_migrate)} test files that need migration:")
    for f in files_to_migrate:
        print(f"  - {f}")
    
    print("\nStarting migration...\n")
    
    for file_path in files_to_migrate:
        migrate_test_file(file_path)
    
    print("\nMigration complete!")
    print("\nNext steps:")
    print("1. Review the migrated files")
    print("2. Update fixtures to use shared ones from conftest.py")
    print("3. Run tests to ensure they still work")
    print("4. Delete the .bak files once confirmed")


if __name__ == '__main__':
    main()