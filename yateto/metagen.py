from .arch import fixArchitectureGlobal
from .codegen.code import Cpp

import os

class MetaGenerator:
    def __init__(self, templateType):
        self.templateType = templateType
        self.generators = []

    def add_generator(self, template, generator, *args, **kwargs):
        assert len(self.templateType) == len(template)
        self.generators += [{
            'name': kwargs["name"] if "name" in kwargs else str(len(self.generators)),
            'template': template,
            'generator': generator,
            'args': args,
            'kwargs': kwargs
        }]

    def compile_list(self, outputDir=''):
        outfiles = []
        for gendata in self.generators:
            outdirname = f'metagen_{gendata["name"]}'
            outdir = os.path.join(outputDir, outdirname)

            genout = []
            for file in ['tensor', 'init', 'kernel', 'test-kernel']:
                genout += [file]
            outfiles += [genout]
        return outfiles

    def generate_single(self, index, outputDir='', namespace='yateto'):
        namespacepfx = 'yatetometagen'
        gendata = self.generators[index]
        subnamespace = f'{namespace}::{namespacepfx}_{gendata["name"]}'
        outdirname = f'metagen_{gendata["name"]}'
        outdir = os.path.join(outputDir, outdirname)
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
                        outdirname = f'metagen_{gendata["name"]}'
                        header.include(f'{outdirname}/{name}.h')
                    with header.Namespace(namespace):
                        for entry in data:
                            self.template(header, entry, data[entry], f'{name}')

        
        headerForward('tensor', tensors)
        headerForward('init', tensors)
        headerForward('kernel', kernels)

        def cppForward(name):
            with Cpp(os.path.join(outputDir, f'{name}.cpp')) as header:
                for gendata in self.generators:
                    outdirname = f'metagen_{gendata["name"]}'
                    header.include(f'{outdirname}/{name}.cpp')
        
        cppForward('tensor')
        cppForward('init')
        cppForward('kernel')
        cppForward('test-kernel')

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
