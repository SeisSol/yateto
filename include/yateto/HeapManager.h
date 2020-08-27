#ifndef YATETO_HEAP_MANAGER_H_
#define YATETO_HEAP_MANAGER_H_

#include <assert.h>

namespace yateto {
    /**
     * \class TmpMemManagerT
     *
     * \brief A naive implementation of stack to handle memory for tmp. variables provided from the user
     *
     * */
    template<typename T>
    struct TmpMemManagerT {
        void attachMem(T* Ptr) {
            m_IsInit = true;
            m_UserSpaceMem = Ptr;
        }

        T* getMem(size_t Size) {
            assert(m_IsInit && "YATETO: Temporary-Memory manager hasn't been initialized");
            int CurrentCount = m_ByteCount;
            m_ByteCount += Size;
            return &m_UserSpaceMem[CurrentCount];
        }

        void flush() {
            m_IsInit = false;
            m_ByteCount = 0;
            m_UserSpaceMem = nullptr;
        }

        private:
        size_t m_ByteCount{0};
        bool m_IsInit{false};
        T *m_UserSpaceMem{nullptr};
    };
}  // YATETO_HEAP_MANAGER_H_
#endif