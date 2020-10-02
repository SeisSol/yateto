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
        void attachMemory(T* ptr) {
            isInit = true;
            userSpaceMem = ptr;
        }

        T* getMem(size_t size) {
            assert(isInit && "YATETO: Temporary-Memory manager hasn't been initialized");
            int currentByteCount = byteCount;
            byteCount += size;
            return &userSpaceMem[currentByteCount];
        }

        void flush() {
            isInit = false;
            byteCount = 0;
            userSpaceMem = nullptr;
        }

        private:
        size_t byteCount{0};
        bool isInit{false};
        T *userSpaceMem{nullptr};
    };
}  // YATETO_HEAP_MANAGER_H_
#endif