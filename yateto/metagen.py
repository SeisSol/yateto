from .arch import fixArchitectureGlobal
from .codegen.code import Cpp

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
    #: What `runtime.h` and `variantOf` define in the runtime namespace, and
    #: which a namespace of tensors therefore cannot be called.
    RUNTIME_NAMES = ('VariantCount', 'VariantNames', 'VariantKeys', 'tensorTable', 'variantOf')

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
        """Per generator name, the host translation units of its code."""
        return collections.OrderedDict(
            (gendata['name'],
             self._paths(gendata, outputDir,
                         self.HOST_SOURCES + ((self.ROUTINES_SOURCE,) if self._owns_routines(gendata) else ())))
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

    def _runtime(self, outputDir, namespace, summaries):
        """`runtime.h` and `runtime.cpp`: every generator, reached by its variant.

        The header names no key and includes no code of a generator, so that
        code which only learns at run time which one it needs does not depend
        on any of them. It says what every variant is and where each one
        stores each tensor. The source reaches the generators through the
        tables they define, and declares those itself for the same reason.
        """
        descriptors = self._tensorDescriptors(summaries)
        for name in descriptors:
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
                header.includeSys('string_view')
                header.include('yateto.h')
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

        with Cpp(os.path.join(outputDir, f'{self.RUNTIME_NAME}.cpp')) as cpp:
            cpp.include(f'{self.RUNTIME_NAME}.h')
            cpp.includeSys('initializer_list')
            cpp.includeSys('stdexcept')
            cpp.includeSys('string')
            with cpp.Namespace(namespace):
                for summary in summaries:
                    subspace = summary['namespace'][len(namespace) + 2:]
                    with cpp.Namespace(subspace), cpp.Namespace(self.INIT_NAMESPACE):
                        cpp.functionDeclaration('tensorTable', '', '::yateto::TensorTable const&')
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
