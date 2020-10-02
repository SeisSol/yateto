#ifndef YATETO_INITTOOLS_H_
#define YATETO_INITTOOLS_H_

#include "CopyPolicy.h"
#include <algorithm>
#include <cstdint>

/*
#ifdef ACL_DEVICE
#include "device.h"
#endif
*/

namespace yateto {
    /** Computes a number of tensors inside of a tensor family.
     *
     * @return a number of tensors.
     * */
    template<class T>
    constexpr size_t numFamilyMembers() {
        return sizeof(T::Size) / sizeof(T::Size[0]);
    }


    /** Computes the next closest aligned memory address for a provided relative address.
     *
     * @param size a pointer address as integer.
     * @param alignment a size of a vector register.
     * @return the next closest aligned relative address.
     * */
    template<typename int_t>
    constexpr size_t alignedUpper(int_t size, size_t alignment) {
        return size + (alignment - size % alignment) % alignment;
    }


    /** Computes a number of real number which fits into a vector register.
     *
     *  NOTE: a size of real number depends of floating number representation i.e. double or float.
     *
     *  @param alignment a size of a vector register in bytes
     *  @return number of real numbers inside of a vector register
     * */
    template<typename float_t>
    constexpr size_t alignedReals(size_t alignment) {
        return alignment / sizeof(float_t);
    }


    /** Computes a size occupied by a tensor family including data alignment b/w tensors in terms of real numbers.
     *
     * NOTE: recursive function.
     *
     * @param alignedReals number of real numbers inside of a vector register.
     * @param n a tensor index inside of a tensor family.
     * @return a size of a tensor family
     * */
    template<class T>
    constexpr size_t computeFamilySize(size_t alignedReals = 1, size_t n = numFamilyMembers<T>()) {
        return n == 0 ? 0 : alignedUpper(T::Size[n-1], alignedReals) + computeFamilySize<T>(alignedReals, n-1);
    }


    template<typename float_t, typename CopyPolicyT>
    class CopyManager {
    public:

        /** Copies data from a tensor to a given memory chunk.
         *
         *  NOTE: The function shifts and aligns a pointer w.r.t. to a given vector register size.
         *
         *  @param mem an address to a chunk of memory.
         *         NOTE: the address is going to be incremented every time
         *         when new information is written.
         *  @param alignment a size of a vector register (in bytes).
         *  @param ptr.
         *  @param alignment.
         * */
        template<class T>
        void copyTensorToMemAndSetPtr(float_t*& mem, float_t*& ptr, size_t alignment = 1) {
            ptr = mem;
            copyValuesToMem(mem, T::Values, T::Values + T::Size, alignment);
        }


        /** Copies data from tensors from a tensor family to a given memory chunk.
         *
         * NOTE: The function writes the actual address (where aligned tensor data stored)
         * back to a tensor family
         *
         *  @param container a reference to a container which contains tensor family data.
         *  @param mem an address to an allocated chunk of memory.
         *         NOTE: the address is going to be incremented every time
         *         when new information is written.
         *  @param alignment a size of a vector register (in bytes).
         * */
        template<class T>
        void copyFamilyToMemAndSetPtr(float_t*& mem,
                                      typename T::template Container<float_t const*>& container,
                                      size_t alignment = 1) {

            // determine a size of the container i.e a number of tensor that it holds
            size_t n = sizeof(T::Size) / sizeof(T::Size[0]);

            for (size_t i = 0; i < n; ++i) {
                // init pointer of each tensor to the allocated memeory
                container.data[i] = mem;

                // copy values and shift pointer
                copyValuesToMem(mem, T::Values[i], T::Values[i] + T::Size[i], alignment);
            }
        }

    protected:
        /** Copies a tensor to a given memory chunk, and shifts a given pointer.
         *
         *  NOTE: The function shifts and aligns a pointer w.r.t. to a given vector register size.
         *
         *  @param mem an address to a chunk of memory.
         *         NOTE: the address is going to be incremented every time
         *         when new information is written.
         *  @param first a pointer to the beginning of tensor data.
         *  @param last a pointer to the end of tensor data.
         *  @param alignment a size of a vector register (in bytes).
         * */
        virtual void copyValuesToMem(float_t*& mem, float_t const* first, float_t const* last, size_t alignment) {

            // copy data
            mem = copier.copy(first, last, mem);

            // shift pointer
            mem += (alignedUpper(reinterpret_cast<uintptr_t>(mem), alignment) - reinterpret_cast<uintptr_t>(mem)) / sizeof(float_t);
            assert(reinterpret_cast<uintptr_t>(mem) % alignment == 0);
        }

    private:
      CopyPolicyT copier{};
    };

    template<class float_t> using DefaultCopyManager = CopyManager<float_t, SimpleCopyPolicy<float_t>>;
}

#endif
