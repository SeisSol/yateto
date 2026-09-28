from .arch import fixArchitectureGlobal
from .codegen.code import Block, Cpp
from .type import Datatype

import collections
import json
import os
import re

class MetaGenerator:
    """Several generators behind one set of names, told apart by a template key.

    Every generator is generated into a namespace and a directory of its own,
    and the headers in the output directory map a template key to the tensors
    and kernels of the generator it was added with: `kernel::X<Key>` is the
    kernel `X` of that generator.

    The code of each generator is compiled in translation units of its own --
    `sources` lists them. Nothing here includes one generator's code into
    another's: a translation unit holding the kernels of every generator
    takes as long to compile as all of them, and no build can spread it over
    more than one core.

    Code that only learns at run time which generator it needs reaches them
    through `runtime.h` instead, which names no key and includes no code of a
    generator. A variant there is the position of a generator in the order it
    was added in, and `runtime::variantOf<Key>()` gives the variant of a key.
    A kernel added with `attrs={'operands': 'runtime'}` is there as well:
    `runtime::kernel::X` takes its operands as views that carry their layout,
    and `execute(variant)` runs the kernel of that variant on them. Each
    generator gets a translation unit of its own that binds the views to its
    kernels -- their own values where they are in the layout the kernel was
    generated for, a copy otherwise -- and its constants from its pool.
    """

    #: Directory, and suffix of the namespace, a generator is generated into.
    DIRECTORY_PREFIX = 'metagen_'
    NAMESPACE_PREFIX = 'yatetometagen_'
    VALID_NAME = r'\w+'

    #: What `Generator.generate` writes for the host, for the device and for
    #: its unit tests, without extension.
    HOST_SOURCES = ('tensor', 'init', 'kernel', 'pool')
    ROUTINES_SOURCE = 'subroutine'
    DEVICE_ROUTINES_SOURCE = 'gpulike_subroutine'
    TEST_SOURCE = 'test-kernel'
    #: What the metagen writes itself, next to the forwarding headers.
    RUNTIME_NAME = 'runtime'
    RUNTIME_NAMESPACE = 'runtime'
    INIT_NAMESPACE = 'init'
    KERNEL_NAMESPACE = 'kernel'
    #: Where each generator binds views to its kernels; a name no tensor and
    #: no kernel can have.
    BINDING_NAMESPACE = '_runtime'
    #: What runs a kernel of `runtime.h`, and which an operand therefore
    #: cannot be called.
    EXECUTE_NAME = 'execute'
    #: What `runtime.h` and `variantOf` define in the runtime namespace, and
    #: which a namespace of tensors therefore cannot be called.
    RUNTIME_NAMES = ('VariantCount', 'VariantNames', 'VariantKeys', 'tensorTable', 'variantOf')
    #: How a scalar of each element type is handed over, whatever type a
    #: variant computes it in.
    SCALAR_TYPES = {'bool': 'bool', 'integer': 'std::int64_t', 'float': 'double'}

    def __init__(self, templateType):
        self.templateType = templateType
        self.generators = []

    def add_generator(self, template, generator, *args, name=None, **kwargs):
        """Adds a generator under a template key.

        The remaining arguments are handed to `Generator.generate`. The name
        is the generator's own: it names the directory and the namespace its
        code goes into, and is not passed on.
        """
        if len(template) != len(self.templateType):
            raise ValueError('A template key needs {} arguments ({}), not {}.'.format(
                len(self.templateType), ', '.join(self.templateType), len(template)))
        name = str(len(self.generators)) if name is None else str(name)
        if not re.fullmatch(self.VALID_NAME, name):
            raise ValueError('A generator name has to be usable as part of an identifier: {}'.format(name))
        if any(gendata['name'] == name for gendata in self.generators):
            raise ValueError('There already is a generator named {}.'.format(name))
        self.generators += [{
            'name': name,
            'template': template,
            'generator': generator,
            'args': args,
            'kwargs': kwargs
        }]

    def _directory(self, gendata, outputDir):
        return os.path.join(outputDir, self.DIRECTORY_PREFIX + gendata['name'])

    def _paths(self, gendata, outputDir, names):
        directory = self._directory(gendata, outputDir)
        return [os.path.join(directory, '{}.cpp'.format(name)) for name in names]

    @staticmethod
    def _owns_routines(gendata):
        # A generator handed a routine cache leaves its routines to whoever
        # holds the cache, and they are written once for all of them there.
        return gendata['kwargs'].get('routine_cache') is None

    def sources(self, outputDir=''):
        """Per generator name, the host translation units of its code.

        Among them the one that binds views to its kernels for `runtime.h`.
        """
        return collections.OrderedDict(
            (gendata['name'],
             self._paths(gendata, outputDir,
                         self.HOST_SOURCES + ((self.ROUTINES_SOURCE,) if self._owns_routines(gendata) else ()) +
                         (self.RUNTIME_NAME,)))
            for gendata in self.generators)

    def device_sources(self, outputDir=''):
        """Per generator name, the translation units a device compiler builds."""
        return collections.OrderedDict(
            (gendata['name'],
             self._paths(gendata, outputDir,
                         (self.DEVICE_ROUTINES_SOURCE,) if self._owns_routines(gendata) else ()))
            for gendata in self.generators)

    def tests(self, outputDir=''):
        """Per generator name, the translation units of its generated unit tests.

        The CxxTest header `KernelTest.t.h` lies next to each of them.
        """
        return collections.OrderedDict(
            (gendata['name'], self._paths(gendata, outputDir, (self.TEST_SOURCE,)))
            for gendata in self.generators)

    def shared_sources(self, outputDir=''):
        """The translation units that belong to no generator: those of `runtime.h`."""
        return [os.path.join(outputDir, '{}.cpp'.format(self.RUNTIME_NAME))]

    def generate_single(self, index, outputDir='', namespace='yateto'):
        """Generates one generator and reports what the metagen needs to know about it.

        The report is plain data, which a build that generates every generator
        in a process of its own can hand to `generate` as `precompiled`.
        """
        gendata = self.generators[index]
        subnamespace = f'{namespace}::{self.NAMESPACE_PREFIX}{gendata["name"]}'
        outdir = self._directory(gendata, outputDir)
        os.makedirs(outdir, exist_ok=True)

        generator = gendata['generator']
        args = gendata['args']
        kwargs = gendata['kwargs']

        fixArchitectureGlobal(generator.arch())
        result = generator.generate(*args, **kwargs, namespace=subnamespace, outputDir=outdir)

        return {
            'name': gendata['name'],
            'namespace': subnamespace,
            'template': [str(part) for part in gendata['template']],
            'tensors': sorted(result['tensors']),
            'kernels': sorted(result['kernels']),
            'tensorTable': [[name, rank] for name, rank in result['tensorTable']],
            'runtime': result['runtime'],
        }

    @staticmethod
    def _guard(namespace, name):
        return re.sub(r'\W', '_', f'METAGEN_{namespace}_{name}_H_'.upper())

    def generate(self, outputDir='', namespace='yateto', includes=[], declarationsTensors=[], declarationsKernels=[], precompiled=None):
        tensors = {}
        kernels = {}

        for tensor in declarationsTensors:
            tensors[tensor] = []
        for tensor in declarationsKernels:
            kernels[tensor] = []

        summaries = []
        for index in range(len(self.generators)):
            if precompiled is None:
                summary = self.generate_single(index, outputDir, namespace)
            else:
                summary = precompiled[index]
            summaries.append(summary)

            for tensor in summary['tensors']:
                tensors.setdefault(tensor, []).append((summary['namespace'], summary['template']))
            for kernel in summary['kernels']:
                kernels.setdefault(kernel, []).append((summary['namespace'], summary['template']))

        def headerForward(name, data, extra=None):
            with Cpp(os.path.join(outputDir, f'{name}.h')) as header:
                with header.HeaderGuard(self._guard(namespace, name)):
                    if extra is not None:
                        header.includeSys('cstddef')
                    for path in includes:
                        header.include(path)
                    for gendata in self.generators:
                        outdirname = self.DIRECTORY_PREFIX + gendata['name']
                        header.include(f'{outdirname}/{name}.h')
                    with header.Namespace(namespace):
                        for entry in data:
                            self.template(header, entry, data[entry], f'{name}')
                        if extra is not None:
                            extra(header)

        headerForward('tensor', tensors, lambda header: self._variantOf(header, summaries))
        headerForward('init', tensors)
        headerForward('kernel', kernels)
        self._runtime(outputDir, namespace, summaries)

    @staticmethod
    def _literal(text):
        return json.dumps(text)

    def _variantOf(self, header, summaries):
        """The variant of a template key, for code that holds the key."""
        templatetypes = ', '.join(f'{typ} Arg{i}' for i, typ in enumerate(self.templateType))
        templateargs = ', '.join(f'Arg{i}' for i, _ in enumerate(self.templateType))
        with header.Namespace(self.RUNTIME_NAMESPACE):
            with header.Namespace('internal'):
                header(f'template<{templatetypes}> struct VariantOf;')
                for variant, summary in enumerate(summaries):
                    header('template<> struct VariantOf<{}> {{ static constexpr std::size_t value = {}; }};'.format(
                        ', '.join(summary['template']), variant))
            header('//! The variant of the generator added under a key, as `runtime.h` takes it.')
            with header.Function('variantOf', '', f'template<{templatetypes}> constexpr std::size_t'):
                header(f'return internal::VariantOf<{templateargs}>::value;')

    def _tensorDescriptors(self, summaries):
        """Per tensor, the number of its group indices and its position in each table."""
        tensors = collections.OrderedDict()
        for variant, summary in enumerate(summaries):
            for position, (name, rank) in enumerate(summary['tensorTable']):
                known = tensors.setdefault(name, (rank, [-1] * len(summaries)))
                if known[0] != rank:
                    raise ValueError('{} is a family of {} indices in one generator and of {} in '
                                     'another.'.format(name, known[0], rank))
                known[1][variant] = position
        return collections.OrderedDict(sorted(tensors.items()))

    @staticmethod
    def _scalarKind(datatype):
        if datatype == 'bool':
            return 'bool'
        return 'float' if Datatype[datatype.upper()].isFloat() else 'integer'

    def _runtimeKernels(self, summaries):
        """Per kernel to take views, what `runtime.h` declares for it.

        Put together over the variants: a member is there if any variant has
        a caller hand it over, writable if any variant writes it, and a family
        as large as the largest one.
        """
        kernels = collections.OrderedDict()
        for variant, summary in enumerate(summaries):
            for interface in summary['runtime']:
                space = interface['namespace']
                key = f'{space}::{interface["name"]}' if space else interface['name']
                rank = None if interface['family'] is None else len(interface['family']['stride'])
                merged = kernels.setdefault(key, {'name': interface['name'], 'namespace': space,
                                                  'rank': rank, 'members': collections.OrderedDict(),
                                                  'variants': collections.OrderedDict()})
                if merged['rank'] != rank:
                    raise ValueError('{} is a kernel family of {} indices in one generator and of {} in '
                                     'another.'.format(key, merged['rank'] or 0, rank or 0))
                for operand in interface['operands'] + [dict(scalar, scalar=True) for scalar in interface['scalars']]:
                    if operand['member'] == self.EXECUTE_NAME:
                        raise ValueError('{} takes an operand named {}, which is what runs it in `{}.h`. '
                                         'Rename the operand.'.format(key, self.EXECUTE_NAME, self.RUNTIME_NAME))
                    kind = self._scalarKind(operand['datatype']) if operand.get('scalar') else 'view'
                    member = merged['members'].setdefault(operand['member'], {
                        'name': operand['name'], 'kind': kind, 'groups': set(), 'rank': operand['rank'],
                        'writable': False})
                    if member['name'] != operand['name'] or member['kind'] != kind or member['rank'] != operand['rank']:
                        raise ValueError('{} takes {} as {} in one generator and as {} in another.'.format(
                            key, operand['member'], self._describe(member), self._describe(dict(operand, kind=kind))))
                    member['groups'].update(tuple(group) for group in operand['groups'])
                    member['writable'] = member['writable'] or operand.get('writable', False)
                merged['variants'][variant] = interface
        return collections.OrderedDict(sorted(kernels.items()))

    @staticmethod
    def _describe(member):
        what = 'a view' if member['kind'] == 'view' else f'a {member["kind"]}'
        indices = f' in a family of {member["rank"]} indices' if member['rank'] else ''
        return f'{what} of {member["name"]}{indices}'

    @staticmethod
    def _familyType(element, groups):
        sizes = [max(group[d] for group in groups) + 1 for d in range(len(next(iter(groups))))]
        return '::yateto::Family<{}, {}>'.format(element, ', '.join(str(size) for size in sizes))

    @staticmethod
    def _at(groups):
        return '({})'.format(', '.join(str(g) for g in groups)) if groups else ''

    def _runtimeStruct(self, header, kernel):
        """A kernel of `runtime.h`: what a caller sets, and `execute` to run it."""
        with header.Struct(kernel['name']):
            for member, data in sorted(kernel['members'].items()):
                if not data['groups']:
                    # Every variant binds it from its pool.
                    continue
                if data['kind'] == 'view':
                    element = '::yateto::RuntimeView' if data['writable'] else '::yateto::ConstRuntimeView'
                    initial = ''
                else:
                    element = self.SCALAR_TYPES[data['kind']]
                    initial = ' = std::numeric_limits<double>::signaling_NaN()' if data['kind'] == 'float' else '{}'
                if data['rank'] > 0:
                    header('{} {};'.format(self._familyType(element, data['groups']), member))
                else:
                    header('{} {}{};'.format(element, member, initial))
            header.emptyline()
            header('//! Runs the kernel of a variant on what is set here.')
            header('void {}({}) const;'.format(self.EXECUTE_NAME, self._executeArgs(kernel)))

    @staticmethod
    def _executeArgs(kernel):
        return 'std::size_t variant' + ''.join(', unsigned i{}'.format(i) for i in range(kernel['rank'] or 0))

    def _bindings(self, outputDir, namespace, summaries, kernels):
        """Per generator, a translation unit that binds views to its kernels.

        Compiled with the generator's own code, since it names its kernels and
        its tensors. A constant comes from the pool, a view is handed to the
        kernel as it is where it is in the layout the kernel was generated
        for, and as a copy otherwise, which is written back after the kernel
        ran where the kernel writes it.
        """
        for variant, (gendata, summary) in enumerate(zip(self.generators, summaries)):
            generated = summary['namespace']
            bound = [(key, kernel, kernel['variants'][variant])
                     for key, kernel in kernels.items() if variant in kernel['variants']]
            with Cpp(os.path.join(self._directory(gendata, outputDir), f'{self.RUNTIME_NAME}.cpp')) as cpp:
                if not bound:
                    # Listed with the sources all the same, so that a build
                    # need not know which generators have such kernels.
                    cpp('// No kernel of {} takes views.'.format(gendata['name']))
                    continue
                cpp.include(f'../{self.RUNTIME_NAME}.h')
                cpp.include('init.h')
                cpp.include('kernel.h')
                cpp.include('pool.h')
                cpp.include('tensor.h')
                cpp.include('yateto/RuntimeView.h')
                cpp.includeSys('stdexcept')
                cpp.includeSys('string')
                for key, kernel, interface in bound:
                    space = kernel['namespace']
                    qualified = '::{}{}::{}::{}'.format(generated, f'::{space}' if space else '',
                                                        self.KERNEL_NAMESPACE, kernel['name'])
                    runtime = '::{}::{}{}::{}::{}'.format(namespace, self.RUNTIME_NAMESPACE, f'::{space}' if space else '',
                                                          self.KERNEL_NAMESPACE, kernel['name'])
                    rank = kernel['rank'] or 0
                    indices = ', '.join(f'i{i}' for i in range(rank))
                    arguments = 'const {}& args'.format(runtime) + ''.join(f', unsigned i{i}' for i in range(rank))
                    with cpp.Namespace(f'{generated}::{self.BINDING_NAMESPACE}'), cpp.Namespace(space), \
                            cpp.Namespace(self.KERNEL_NAMESPACE):
                        with cpp.Function(kernel['name'], arguments):
                            if interface['family'] is not None:
                                stride = interface['family']['stride']
                                size = interface['family']['size']
                                # One extent per index, the last one as far as the family reaches.
                                extents = [stride[d + 1] // stride[d] for d in range(rank - 1)] + \
                                    [-(-size // stride[-1])]
                                cpp('const unsigned position = {};'.format(
                                    ' + '.join('{}u * i{}'.format(st, i) for i, st in enumerate(stride))))
                                with cpp.If(' || '.join(['i{} >= {}u'.format(i, extent) for i, extent in enumerate(extents)] +
                                                        ['position >= {}u'.format(size),
                                                         '{}::findExecute({}) == nullptr'.format(qualified, indices)])):
                                    cpp('throw std::out_of_range("{}({}) is no kernel of the family in variant {}.");'.format(
                                        key, ', '.join('" + std::to_string(i{}) + "'.format(i) for i in range(rank)),
                                        gendata['name']))
                            cpp('static const ::{0}::Pool pool = ::{0}::Pool::host();'.format(generated))
                            cpp(f'{qualified} krnl;')
                            cpp('krnl.bindGlobals(pool);')
                            for scalar in interface['scalars']:
                                datatype = Datatype[scalar['datatype'].upper()]
                                for group in scalar['groups']:
                                    at = self._at(group)
                                    value = f'args.{scalar["member"]}{at}'
                                    if datatype in (Datatype.F16, Datatype.BF16):
                                        # Not every compiler converts a double to these directly.
                                        value = f'static_cast<float>({value})'
                                    cpp('krnl.{}{} = static_cast<{}>({});'.format(scalar['member'], at, datatype.ctype(), value))
                            # Each member of a family with what it uses, so that
                            # a caller only hands over that; the members that
                            # use the same take the same way in.
                            ways = collections.OrderedDict()
                            for entry in interface['kernels']:
                                ways.setdefault(json.dumps([entry['uses'], entry['writes']], sort_keys=True),
                                                []).append(entry)
                            if len(ways) == 1:
                                self._bindKernel(cpp, gendata, generated, key, interface, interface['kernels'][0], indices)
                            else:
                                with Block(cpp, 'switch (position)'):
                                    for entries in ways.values():
                                        for entry in entries[:-1]:
                                            cpp('case {}:'.format(entry['position']))
                                        with Block(cpp, 'case {}:'.format(entries[-1]['position'])):
                                            self._bindKernel(cpp, gendata, generated, key, interface, entries[0], indices)
                                            cpp('break;')
                                    cpp('default:')
                                    cpp('  break;')

    def _bindKernel(self, cpp, gendata, generated, key, interface, entry, indices):
        """Hands a kernel what one member of it uses, runs it, and writes back what it wrote."""
        written = []
        count = 0
        for operand in interface['operands']:
            ctype = Datatype[operand['datatype'].upper()].ctype()
            prefix, _, name = operand['name'].rpartition('::')
            init = '::{}{}::{}::{}'.format(generated, f'::{prefix}' if prefix else '', self.INIT_NAMESPACE, name)
            for group in entry['uses'].get(operand['name'], []):
                at = self._at(group)
                variable = 'operand{}'.format(count)
                count += 1
                what = self._literal('{}{} of {} in variant {}'.format(operand['member'], at, key, gendata['name']))
                cpp('::yateto::Operand<{}> {}(args.{}{}, *{}::descriptor({}), {});'.format(
                    ctype, variable, operand['member'], at, init, ', '.join(str(g) for g in group), what))
                cpp('krnl.{}{} = {}.data();'.format(operand['member'], at, variable))
                if operand['name'] in entry['writes']:
                    written.append(variable)
        cpp('krnl.execute({});'.format(indices))
        for variable in written:
            cpp(f'{variable}.finish();')

    def _runtime(self, outputDir, namespace, summaries):
        """`runtime.h` and `runtime.cpp`: every generator, reached by its variant.

        The header names no key and includes no code of a generator, so that
        code which only learns at run time which one it needs does not depend
        on any of them. It says what every variant is and where each one
        stores each tensor. The source reaches the generators through the
        tables they define, and declares those itself for the same reason.
        """
        descriptors = self._tensorDescriptors(summaries)
        kernels = self._runtimeKernels(summaries)
        for name in list(descriptors) + list(kernels):
            space = name.split('::')[0] if '::' in name else None
            if space in self.RUNTIME_NAMES:
                raise ValueError('The namespace of {} is also a name `runtime.h` defines, '
                                 '{}::{}. Rename the namespace.'.format(name, self.RUNTIME_NAMESPACE, space))
        count = len(summaries)
        variantArg = 'std::size_t variant'

        def groupArgs(rank):
            return ''.join(', unsigned i{}'.format(i) for i in range(rank))

        def byNamespace(names):
            spaces = collections.OrderedDict()
            for name in names:
                prefix, _, base = name.rpartition('::')
                spaces.setdefault(prefix, []).append(base)
            return spaces

        with Cpp(os.path.join(outputDir, f'{self.RUNTIME_NAME}.h')) as header:
            with header.HeaderGuard(self._guard(namespace, self.RUNTIME_NAME)):
                header.includeSys('cstddef')
                header.includeSys('cstdint')
                header.includeSys('limits')
                header.includeSys('string_view')
                header.include('yateto.h')
                header.include('yateto/RuntimeView.h')
                with header.Namespace(namespace), header.Namespace(self.RUNTIME_NAMESPACE):
                    header('//! How many generators there are; a variant is the position of one of them.')
                    header(f'constexpr std::size_t VariantCount = {count};')
                    if count > 0:
                        header('//! Per variant, the name its generator was added under.')
                        header('constexpr std::string_view VariantNames[] = {{{}}};'.format(
                            ', '.join(self._literal(summary['name']) for summary in summaries)))
                        header('//! Per variant, the template key of its generator.')
                        header('constexpr std::string_view VariantKeys[] = {{{}}};'.format(
                            ', '.join(self._literal(', '.join(summary['template'])) for summary in summaries)))
                    header('//! Every tensor of a variant, ordered by name.')
                    header.functionDeclaration('tensorTable', variantArg, '::yateto::TensorTable const&')
                    for prefix, names in byNamespace(descriptors).items():
                        with header.Namespace(prefix), header.Namespace(self.INIT_NAMESPACE):
                            for name in names:
                                rank, _ = descriptors[f'{prefix}::{name}' if prefix else name]
                                with header.Struct(name):
                                    header('//! Where a variant stores {}; null where it has none.'.format(name))
                                    header.functionDeclaration('descriptor', variantArg + groupArgs(rank),
                                                               'static ::yateto::TensorDescriptor const*')
                    for space, names in byNamespace(kernels).items():
                        with header.Namespace(space), header.Namespace(self.KERNEL_NAMESPACE):
                            for name in names:
                                self._runtimeStruct(header, kernels[f'{space}::{name}' if space else name])

        with Cpp(os.path.join(outputDir, f'{self.RUNTIME_NAME}.cpp')) as cpp:
            cpp.include(f'{self.RUNTIME_NAME}.h')
            cpp.includeSys('initializer_list')
            cpp.includeSys('stdexcept')
            cpp.includeSys('string')
            with cpp.Namespace(namespace):
                for variant, summary in enumerate(summaries):
                    subspace = summary['namespace'][len(namespace) + 2:]
                    with cpp.Namespace(subspace), cpp.Namespace(self.INIT_NAMESPACE):
                        cpp.functionDeclaration('tensorTable', '', '::yateto::TensorTable const&')
                    for key, kernel in kernels.items():
                        if variant not in kernel['variants']:
                            continue
                        runtime = '::{}::{}{}::{}::{}'.format(
                            namespace, self.RUNTIME_NAMESPACE, f'::{kernel["namespace"]}' if kernel['namespace'] else '',
                            self.KERNEL_NAMESPACE, kernel['name'])
                        with cpp.Namespace(f'{subspace}::{self.BINDING_NAMESPACE}'), \
                                cpp.Namespace(kernel['namespace']), cpp.Namespace(self.KERNEL_NAMESPACE):
                            cpp.functionDeclaration(kernel['name'], 'const {}& args{}'.format(
                                runtime, ''.join(f', unsigned i{i}' for i in range(kernel['rank'] or 0))))
                with cpp.Namespace(self.RUNTIME_NAMESPACE):
                    # Under a name no tensor can have, since a tensor may be
                    # named like any of these.
                    cpp('namespace _detail {')
                    cpp('namespace {')
                    cpp('using TableFunction = ::yateto::TensorTable const& (*)();')
                    if count > 0:
                        cpp('constexpr TableFunction Tables[] = {{{}}};'.format(', '.join(
                            '&::{}::{}::tensorTable'.format(summary['namespace'], self.INIT_NAMESPACE)
                            for summary in summaries)))
                    with cpp.Function('checkVariant', variantArg):
                        with cpp.If('variant >= VariantCount'):
                            cpp('throw std::out_of_range("There is no variant " + std::to_string(variant) + "; '
                                'there are " + std::to_string(VariantCount) + ".");')
                    with cpp.Function('member', variantArg + ', std::ptrdiff_t const* positions, '
                                      'std::initializer_list<unsigned> group',
                                      '::yateto::TensorDescriptor const*'):
                        cpp('checkVariant(variant);')
                        with cpp.If('positions[variant] < 0'):
                            cpp('return nullptr;')
                        if count > 0:
                            cpp('return Tables[variant]().entries[positions[variant]].member(group);')
                        else:
                            cpp('return nullptr;')
                    cpp('} // namespace')
                    cpp('} // namespace _detail')
                    cpp.emptyline()
                    with cpp.Function('tensorTable', variantArg, '::yateto::TensorTable const&'):
                        cpp('_detail::checkVariant(variant);')
                        if count > 0:
                            cpp('return _detail::Tables[variant]();')
                        else:
                            cpp('static constexpr ::yateto::TensorTable Empty{nullptr, 0};')
                            cpp('return Empty;')
                    for prefix, names in byNamespace(descriptors).items():
                        with cpp.Namespace(prefix), cpp.Namespace(self.INIT_NAMESPACE):
                            for name in names:
                                rank, positions = descriptors[f'{prefix}::{name}' if prefix else name]
                                with cpp.Function(f'{name}::descriptor', variantArg + groupArgs(rank),
                                                  '::yateto::TensorDescriptor const*'):
                                    cpp('static constexpr std::ptrdiff_t Positions[] = {{{}}};'.format(
                                        ', '.join(str(position) for position in positions)))
                                    cpp('return ::{}::{}::_detail::member(variant, Positions, {{{}}});'.format(
                                        namespace, self.RUNTIME_NAMESPACE, ', '.join(f'i{i}' for i in range(rank))))
                    for space, names in byNamespace(kernels).items():
                        with cpp.Namespace(space), cpp.Namespace(self.KERNEL_NAMESPACE):
                            for name in names:
                                key = f'{space}::{name}' if space else name
                                kernel = kernels[key]
                                rank = kernel['rank'] or 0
                                with cpp.Function(f'{name}::{self.EXECUTE_NAME}', self._executeArgs(kernel), 'void',
                                                  const=True):
                                    cpp('using Function = void (*)(const {}&{});'.format(
                                        name, ', unsigned' * rank))
                                    cpp('static constexpr Function Functions[] = {{{}}};'.format(', '.join(
                                        '&::{}::{}{}::{}::{}'.format(
                                            summary['namespace'], self.BINDING_NAMESPACE, f'::{space}' if space else '',
                                            self.KERNEL_NAMESPACE, name)
                                        if variant in kernel['variants'] else 'nullptr'
                                        for variant, summary in enumerate(summaries))))
                                    cpp('::{}::{}::_detail::checkVariant(variant);'.format(namespace, self.RUNTIME_NAMESPACE))
                                    with cpp.If('Functions[variant] == nullptr'):
                                        cpp('throw std::invalid_argument("The variant " + std::string(::{}::{}::VariantNames[variant]) + '
                                            '" has no kernel {}.");'.format(namespace, self.RUNTIME_NAMESPACE, key))
                                    cpp('Functions[variant](*this{});'.format(''.join(f', i{i}' for i in range(rank))))
        self._bindings(outputDir, namespace, summaries, kernels)

    def namespacing(self, header, spaces, inner):
        if len(spaces) == 0:
            inner()
        else:
            with header.Namespace(spaces[0]):
                self.namespacing(header, spaces[1:], inner)

    def template(self, header, prename, foundin, subnsp):
        splitname = prename.split('::')

        assert len(splitname) > 0

        def inner():
            name = splitname[-1]
            fullname = '::'.join(splitname[:-1] + [subnsp, splitname[-1]])
            escname = name.replace(':', '_')
            internalName = f'Internal_{escname}'

            templatetypes = ', '.join(f'{typ} Arg{i}' for i, typ in enumerate(self.templateType))
            templateargs = ', '.join(f'Arg{i}' for i, _ in enumerate(self.templateType))

            with header.Namespace('internal'):
                header(f'template<{templatetypes}> struct {internalName} {"{"} using Type = void; {"}"};')
                for gnsp, spec in foundin:
                    spectext = ', '.join(str(specpart) for specpart in spec)
                    header(f'template<> struct {internalName}<{spectext}> {"{"} using Type = ::{gnsp}::{fullname}; {"}"};')
            header(f'template<{templatetypes}> using {name} = typename internal::{internalName}<{templateargs}>::Type;')

        self.namespacing(header, splitname[:-1] + [subnsp], inner)
