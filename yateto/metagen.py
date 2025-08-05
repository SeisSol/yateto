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
            'template': template,
            'generator': generator,
            'args': args,
            'kwargs': kwargs
        }]
    
    def generate(self, outputDir='', namespace='yateto', includes=[], declarationsTensors=[], declarationsKernels=[]):
        tensors = {}
        kernels = {}

        for tensor in declarationsTensors:
            tensors[tensor] = []
        for tensor in declarationsKernels:
            kernels[tensor] = []

        namespacepfx = 'yatetometagen'
        for i, gendata in enumerate(self.generators):
            subnamespace = f'{namespace}::{namespacepfx}{i}'
            outdir = os.path.join(outputDir, f'metagen{i}')
            os.makedirs(outdir, exist_ok=True)

            generator = gendata['generator']
            template = gendata['template']
            args = gendata['args']
            kwargs = gendata['kwargs']

            fixArchitectureGlobal(generator._arch)
            result = generator.generate(*args, **kwargs, namespace=subnamespace, outputDir=outdir)

            for tensor in result['tensors']:
                if tensor not in tensors:
                    tensors[tensor] = []
                tensors[tensor] += [(subnamespace, template)]
            for tensor in result['kernels']:
                if tensor not in kernels:
                    kernels[tensor] = []
                kernels[tensor] += [(subnamespace, template)]
        
        # TODO: open meta header
        with Cpp(os.path.join(outputDir, 'tensor.h')) as header:
            with header.HeaderGuard('METAGEN_TENSOR_H_'):
                for path in includes:
                    header.include(path)
                for i, gendata in enumerate(self.generators):
                    header.include(f'metagen{i}/tensor.h')
                with header.Namespace(namespace):
                    for tensor in tensors:
                        self.template(header, tensor, tensors[tensor], 'tensor')
        
        with Cpp(os.path.join(outputDir, 'init.h')) as header:
            with header.HeaderGuard('METAGEN_INIT_H_'):
                for path in includes:
                    header.include(path)
                for i, gendata in enumerate(self.generators):
                    header.include(f'metagen{i}/init.h')
                with header.Namespace(namespace):
                    for tensor in tensors:
                        self.template(header, tensor, tensors[tensor], 'init')
        
        with Cpp(os.path.join(outputDir, 'kernel.h')) as header:
            with header.HeaderGuard('METAGEN_KERNEL_H_'):
                for path in includes:
                    header.include(path)
                for i, gendata in enumerate(self.generators):
                    header.include(f'metagen{i}/kernel.h')
                with header.Namespace(namespace):
                    for kernel in kernels:
                        self.template(header, kernel, kernels[kernel], 'kernel')
        
        with Cpp(os.path.join(outputDir, 'tensor.cpp')) as header:
            for i, gendata in enumerate(self.generators):
                header.include(f'metagen{i}/tensor.cpp')
        
        with Cpp(os.path.join(outputDir, 'init.cpp')) as header:
            for i, gendata in enumerate(self.generators):
                header.include(f'metagen{i}/init.cpp')
        
        with Cpp(os.path.join(outputDir, 'kernel.cpp')) as header:
            for i, gendata in enumerate(self.generators):
                header.include(f'metagen{i}/kernel.cpp')
        
        with Cpp(os.path.join(outputDir, 'test-kernels.cpp')) as header:
            for i, gendata in enumerate(self.generators):
                header.include(f'metagen{i}/test-kernels.cpp')
        
    def namespacing(self, header, spaces, inner):
        if len(spaces) == 0:
            inner()
        else:
            with header.Namespace(spaces[0]):
                self.namespacing(header, spaces[1:], inner)

    def template(self, header, prename, foundin, subnsp):
        splitname = prename.split('::')

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
