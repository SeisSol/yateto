from .arch import fixArchitectureGlobal
from .codegen.code import Cpp

import collections
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

    def generate_single(self, index, outputDir='', namespace='yateto'):
        gendata = self.generators[index]
        subnamespace = f'{namespace}::{self.NAMESPACE_PREFIX}{gendata["name"]}'
        outdir = self._directory(gendata, outputDir)
        os.makedirs(outdir, exist_ok=True)

        generator = gendata['generator']
        template = gendata['template']
        args = gendata['args']
        kwargs = gendata['kwargs']

        fixArchitectureGlobal(generator.arch())
        result = generator.generate(*args, **kwargs, namespace=subnamespace, outputDir=outdir)

        tensors = {}
        kernels = {}

        for tensor in result['tensors']:
            tensors[tensor] = (subnamespace, template)
        for kernel in result['kernels']:
            kernels[kernel] = (subnamespace, template)

        return tensors, kernels

    def generate(self, outputDir='', namespace='yateto', includes=[], declarationsTensors=[], declarationsKernels=[], precompiled=None):
        tensors = {}
        kernels = {}

        for tensor in declarationsTensors:
            tensors[tensor] = []
        for tensor in declarationsKernels:
            kernels[tensor] = []

        for index in range(len(self.generators)):
            if precompiled is None:
                local_tensors, local_kernels = self.generate_single(index, outputDir, namespace)
            else:
                local_tensors, local_kernels = precompiled[index]

            for tensor in local_tensors:
                if tensor not in tensors:
                    tensors[tensor] = []
                tensors[tensor] += [local_tensors[tensor]]
            for kernel in local_kernels:
                if kernel not in kernels:
                    kernels[kernel] = []
                kernels[kernel] += [local_kernels[kernel]]

        nspuppercase = namespace.upper()

        def headerForward(name, data):
            upper = name.upper()
            with Cpp(os.path.join(outputDir, f'{name}.h')) as header:
                with header.HeaderGuard(f'METAGEN_{nspuppercase}_{upper}_H_'):
                    for path in includes:
                        header.include(path)
                    for gendata in self.generators:
                        outdirname = self.DIRECTORY_PREFIX + gendata['name']
                        header.include(f'{outdirname}/{name}.h')
                    with header.Namespace(namespace):
                        for entry in data:
                            self.template(header, entry, data[entry], f'{name}')


        headerForward('tensor', tensors)
        headerForward('init', tensors)
        headerForward('kernel', kernels)

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
