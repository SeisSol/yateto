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
            for kernel in result['kernels']:
                if kernel not in kernels:
                    kernels[kernel] = []
                kernels[kernel] += [(subnamespace, template)]

        nspuppercase = namespace.upper()

        def headerForward(name, data):
            upper = name.upper()
            with Cpp(os.path.join(outputDir, f'{name}.h')) as header:
                with header.HeaderGuard(f'METAGEN_{nspuppercase}_{upper}_H_'):
                    for path in includes:
                        header.include(path)
                    for i, gendata in enumerate(self.generators):
                        header.include(f'metagen{i}/{name}.h')
                    with header.Namespace(namespace):
                        for entry in data:
                            self.template(header, entry, data[entry], f'{name}')

        
        headerForward('tensor', tensors)
        headerForward('init', tensors)
        headerForward('kernel', kernels)

        def cppForward(name):
            with Cpp(os.path.join(outputDir, f'{name}.cpp')) as header:
                for i, gendata in enumerate(self.generators):
                    header.include(f'metagen{i}/{name}.cpp')
        
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
