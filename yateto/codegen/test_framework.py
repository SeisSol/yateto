from abc import ABC, abstractmethod

class TestFramework(ABC):
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
        with cpp.PPIfndef('NDEBUG'):
            cpp('long long libxsmm_num_total_flops = 0;')
            cpp('long long pspamm_num_total_flops = 0;')

class CxxTest(TestFramework):
    TEST_CLASS = 'KernelTestSuite'
    TEST_NAMESPACE = 'unit_test'
    TEST_PREFIX = 'test'

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

    def functionArgs(self, testName):
        """functionArgs.

        :param testName: Name of test
        """
        return {'name': 'SUBCASE', 'arguments': '"{}"'.format(testName), 'returnType': ''}

    def assertLessThan(self, x, y):
        return 'CHECK({} < {});'.format(x, y);

    def generate(self, cpp, namespace, kernelsInclude, initInclude, body):
        super().generate(cpp, namespace, kernelsInclude, initInclude, body)
        cpp.include('doctest.h')
        cpp('using namespace {};'.format(namespace))
        with cpp.Function(name='TEST_CASE', arguments='"{}"'.format(self.TEST_CASE), returnType=''):
            body(cpp, self)
