#!/usr/bin/env python3
"""What a generated kernel set costs, before anyone runs it.

Reads a directory yateto has generated into and reports, per kernel, the
numbers that say how much work it does and how much memory it moves to do it:
the flop counts and byte counts the header states, the temporary storage it
asks for, and, from the generated source, how many passes over an index space
it makes and how much it puts in a buffer for a later statement to read.

Two reports compare. That is what this is for: a change to the code generator
shows up here before a machine is asked about it, and where nothing moved
there is nothing to measure.

    kernel_report.py <generated-dir> -o before.json
    kernel_report.py <generated-dir> -o after.json
    kernel_report.py --compare before.json after.json

The numbers are static. They say what the generator decided, not what the
hardware makes of it -- a kernel that got faster and one that merely got
shorter look the same from here.
"""

import argparse
import json
import pathlib
import re
import sys

#: The constants the generated header states per kernel.
COUNTERS = ['NonZeroFlops', 'HardwareFlops', 'InboundConstBytes', 'InboundBytes',
            'OutboundBytes', 'TmpMemRequiredInBytes', 'TmpMaxMemRequiredInBytes']

_STRUCT = re.compile(r'^\s*struct (\w+) \{')
_COUNTER = re.compile(r'^\s*constexpr static unsigned long const (\w+) = (\d+);')
_COUNTERS = re.compile(r'^\s*constexpr static unsigned long const (\w+)\[\] = \{([\d, ]*)\};')
_EXECUTE = re.compile(r'^\s*void ([\w:]+)::execute\d*\(\) \{')


def counters(header):
    """Per kernel, the constants the header states.

    A family states one number per member, as an array; the members are
    summed, since what a caller pays for the family is what its members cost
    together.
    """
    kernels = {}
    name = None
    for line in header.splitlines():
        struct = _STRUCT.match(line)
        if struct:
            name = struct.group(1)
            kernels.setdefault(name, {counter: 0 for counter in COUNTERS})
            continue
        counter = _COUNTER.match(line)
        if counter and name and counter.group(1) in COUNTERS:
            kernels[name][counter.group(1)] += int(counter.group(2))
            continue
        family = _COUNTERS.match(line)
        if family and name and family.group(1) in COUNTERS:
            # a family states one number per member; what the caller pays for
            # the family is what its members cost together
            kernels[name][family.group(1)] += sum(
                int(value) for value in family.group(2).split(',') if value.strip())
    return kernels


def bodies(source):
    """Per kernel, the body of its execute()."""
    lines = source.splitlines()
    found = {}
    for position, line in enumerate(lines):
        execute = _EXECUTE.match(line)
        if not execute:
            continue
        # a family states one execute per member, all on the same struct
        name = execute.group(1).split('::')[-1]
        depth = 0
        body = []
        for line in lines[position:]:
            depth += line.count('{') - line.count('}')
            body.append(line)
            if depth == 0:
                break
        found.setdefault(name, []).extend(body)
    return found


def shape(body):
    """What the generated body does, counted.

    `nests` is how many times the kernel walks an index space; `buffers` is
    how many temporaries it keeps, which is what it puts in memory for a later
    statement to read; `calls` is how much of the work it hands to a generated
    routine rather than doing itself.
    """
    text = '\n'.join(body)
    return {
        'nests': text.count('#pragma omp simd'),
        'loops': len(re.findall(r'\bfor \(', text)),
        'buffers': len(set(re.findall(r'\b(_tmp\d+)\b', text))),
        'memsets': text.count('memset('),
        'calls': len(re.findall(r'^\s*\w*(?:gemm|sparse|dense)\w*\(', text,
                                re.MULTILINE | re.IGNORECASE)),
        'lines': len([line for line in body if line.strip()]),
    }


def report(directory):
    directory = pathlib.Path(directory)
    header = (directory / 'kernel.h')
    source = (directory / 'kernel.cpp')
    if not header.exists() or not source.exists():
        raise SystemExit(f'{directory} holds no generated kernel.h and kernel.cpp')

    kernels = counters(header.read_text())
    generated = bodies(source.read_text())
    for name, body in generated.items():
        kernels.setdefault(name, {counter: 0 for counter in COUNTERS})
        kernels[name].update(shape(body))
    # a struct the header states but the source never defines is a kernel
    # family's base, which has no body of its own
    return {name: values for name, values in kernels.items() if name in generated}


def compare(before, after, only=None):
    """The kernels whose numbers moved, and by how much."""
    names = sorted(set(before) | set(after))
    fields = COUNTERS + ['nests', 'loops', 'buffers', 'memsets', 'calls', 'lines']
    rows = []
    for name in names:
        one, two = before.get(name, {}), after.get(name, {})
        changed = {field: (one.get(field), two.get(field))
                   for field in fields
                   if one.get(field) != two.get(field)
                   and (only is None or field in only)}
        if changed:
            rows.append((name, changed))
    return rows


def totals(kernels, fields):
    return {field: sum(values.get(field, 0) or 0 for values in kernels.values())
            for field in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('directory', nargs='?',
                        help='a directory yateto generated into')
    parser.add_argument('-o', '--output', help='write the report here as JSON')
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'),
                        help='two reports to compare')
    parser.add_argument('--only', nargs='+', metavar='FIELD',
                        help='report only these fields')
    args = parser.parse_args()

    if args.compare:
        before = json.loads(pathlib.Path(args.compare[0]).read_text())
        after = json.loads(pathlib.Path(args.compare[1]).read_text())
        rows = compare(before, after, args.only)
        if not rows:
            print('nothing moved')
            return 0
        width = max(len(name) for name, _ in rows)
        for name, changed in rows:
            parts = ', '.join(f'{field} {one} -> {two}'
                              for field, (one, two) in sorted(changed.items()))
            print(f'{name:<{width}}  {parts}')
        fields = args.only or (COUNTERS + ['nests', 'loops', 'buffers',
                                           'memsets', 'calls', 'lines'])
        one, two = totals(before, fields), totals(after, fields)
        moved = [f'{field} {one[field]} -> {two[field]}'
                 for field in fields if one[field] != two[field]]
        print(f'\n{len(rows)} of {len(set(before) | set(after))} kernels moved')
        if moved:
            print('total: ' + ', '.join(moved))
        return 0

    if not args.directory:
        parser.error('give a directory to report on, or --compare two reports')

    kernels = report(args.directory)
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(kernels, indent=2,
                                                        sort_keys=True))
        return 0

    fields = args.only or ['HardwareFlops', 'TmpMemRequiredInBytes', 'nests',
                           'buffers', 'lines']
    width = max([len(name) for name in kernels] + [6])
    print(f'{"kernel":<{width}}  ' + '  '.join(f'{field:>12}' for field in fields))
    for name in sorted(kernels):
        values = kernels[name]
        print(f'{name:<{width}}  '
              + '  '.join(f'{values.get(field, 0):>12}' for field in fields))
    summed = totals(kernels, fields)
    print(f'{"total":<{width}}  '
          + '  '.join(f'{summed[field]:>12}' for field in fields))
    return 0


if __name__ == '__main__':
    sys.exit(main())
