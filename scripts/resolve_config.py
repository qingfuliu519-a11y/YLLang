#!/usr/bin/env python3
"""
Convert a YAML config file into a C++ header with constexpr constants.

Usage:
    python resolve_config.py <input.yaml> [output.h]
"""

import sys
import os
import yaml
import re


def to_camel_case(*parts):
    """Convert path parts to kCamelCase variable name."""
    camel_parts = [part[0].upper() + part[1:].lower() for part in parts]
    return 'k' + ''.join(camel_parts)


def gather_constants(node, path, constants):
    """Recursively collect scalar values from YAML node."""
    if isinstance(node, dict):
        for key, value in node.items():
            gather_constants(value, path + [key], constants)
    elif isinstance(node, list):
        # Lists are skipped as per requirement (can be extended)
        pass
    else:
        # Scalar values
        var_name = to_camel_case(*path)

        if isinstance(node, bool):
            cpp_type = 'bool'
            value_str = 'true' if node else 'false'
        elif isinstance(node, int):
            cpp_type = 'int'
            value_str = str(node)
        elif isinstance(node, float):
            cpp_type = 'double'
            value_str = repr(node)
        elif isinstance(node, str):
            cpp_type = 'std::string'
            escaped = node.replace('\\', '\\\\').replace('"', '\\"')
            value_str = f'"{escaped}"'
        elif node is None:
            return
        else:
            return

        constants.append((var_name, cpp_type, value_str))


def write_header(output_path, constants):
    """Write the C++ header file."""
    with open(output_path, 'w') as f:
        f.write("""// Auto-generated from YAML configuration. DO NOT EDIT.

#ifndef YLLANG_CONFIG_H_
#define YLLANG_CONFIG_H_

#include <string>

namespace yllang {

""")
        for var_name, cpp_type, value_str in constants:
            f.write(f"constexpr {cpp_type} {var_name} = {value_str};\n")

        f.write("""
} // namespace yllang

#endif // YLLANG_CONFIG_H_
""")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    # Derive output path if not provided
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = base + '.h'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    constants = []
    gather_constants(data, [], constants)
    constants.sort(key=lambda x: x[0])  # for consistent output

    write_header(output_path, constants)
    return 0

if __name__ == '__main__':
    main()
