from abc import ABC, abstractmethod

class TestFramework(ABC):
    def __init__(self, arch):
        self.arch = arch

    @abstractmethod
    def functionArgs(self, testName):
        """functionArgs.

        :param testName: Name of test
        """
        pass

    @abstractmethod
    def assertLessThan(self, x, y):
        """Should return code which checks x < y."""
        pass

    @abstractmethod
    def generate(self, cpp, namespace, kernelsInclude, initInclude, body):
        """generate unit test file for cxxtest.

        :param cpp: code.Cpp object
        :param namespace: Namespace string
        :param kernelsInclude: Kernels header file
        :param initInclude: Init header File
        :param body: Function which accepts cpp and self
        """
        cpp.include(kernelsInclude)
        cpp.include(initInclude)
        cpp.include('yateto.h')
        # what the test bodies call themselves: sqrt, the aligned operator new/delete, memset
        cpp.includeSys('cmath')
        cpp.includeSys('cstdlib')
        cpp.includeSys('cstring')
        cpp.includeSys('new')
        for header in self.arch.headers():
            cpp.includeSys(header)
        with cpp.PPIfndef('NDEBUG'):
            with cpp.PPIfndef('YATETO_TESTING_NO_FLOP_COUNTER'):
                cpp('long long libxsmm_num_total_flops = 0;')
                cpp('long long pspamm_num_total_flops = 0;')

class CxxTest(TestFramework):
    TEST_CLASS = 'KernelTestSuite'
    TEST_NAMESPACE = 'unit_test'
    TEST_PREFIX = 'test'

    def __init__(self, arch):
        super().__init__(arch)

    def functionArgs(self, testName):
        return {'name': self.TEST_PREFIX + testName}

    def assertLessThan(self, x, y):
        return 'TS_ASSERT_LESS_THAN({}, {});'.format(x, y);

    def generate(self, cpp, namespace, kernelsInclude, initInclude, body):
        super().generate(cpp, namespace, kernelsInclude, initInclude, body)
        cpp.includeSys('cxxtest/TestSuite.h')
        with cpp.Namespace(namespace):
            with cpp.Namespace(self.TEST_NAMESPACE):
                cpp.classDeclaration(self.TEST_CLASS)
        with cpp.Class('{}::{}::{} : public CxxTest::TestSuite'.format(namespace, self.TEST_NAMESPACE, self.TEST_CLASS)):
            cpp.label('public')
            body(cpp, self)

class Doctest(TestFramework):
    TEST_CASE = 'yateto kernels'
    #: The namespace, in that of the kernels, of the functions the tests are.
    TEST_NAMESPACE = 'unit_test'

    def __init__(self, arch):
        super().__init__(arch)
        self._tests = []

    def functionArgs(self, testName):
        """The function the test of a kernel is written as.

        The test of every kernel is a function of its own, which the test case
        runs as a subcase. Written as the subcases themselves, the tests of all
        kernels are one function, which a compiler takes several times as long
        for as for the functions it is made of, at a multiple of the memory.

        :param testName: Name of test
        """
        self._tests.append(testName)
        return {'name': testName}

    def assertLessThan(self, x, y):
        return 'CHECK({} < {});'.format(x, y);

    def generate(self, cpp, namespace, kernelsInclude, initInclude, body):
        super().generate(cpp, namespace, kernelsInclude, initInclude, body)
        cpp.include('doctest.h')
        self._tests = []
        with cpp.Namespace(namespace):
            with cpp.Namespace(self.TEST_NAMESPACE):
                body(cpp, self)
        with cpp.Function(name='TEST_CASE', arguments=f'"{self.TEST_CASE} for \\"{namespace}\\""', returnType=''):
            for test in self._tests:
                cpp(f'SUBCASE("{test}") {{ {namespace}::{self.TEST_NAMESPACE}::{test}(); }}')
