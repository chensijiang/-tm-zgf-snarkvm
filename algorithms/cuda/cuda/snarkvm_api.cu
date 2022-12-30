// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

#include <cuda.h>
#include <chrono>
#include "snarkvm.cu"

#include <iostream>

#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>


#ifndef __CUDA_ARCH__

template<typename T>
class threadsafe_queue {
private:
    mutable std::mutex mut;
    std::queue<T> data_queue;
    std::condition_variable data_cond;
public:
    threadsafe_queue(){}
    threadsafe_queue(threadsafe_queue const& other) {
        std::lock_guard<std::mutex> lk(other.mut);
        data_queue=other.data_queue;
    }

    void push(T new_value) {
        std::lock_guard<std::mutex> lk(mut);
        data_queue.push(new_value);
        data_cond.notify_one();
    }

    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk,[this]{return !data_queue.empty();});
        value=data_queue.front();
        data_queue.pop();
    }

    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk,[this]{return !data_queue.empty();});
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mut);
        if(data_queue.empty())
            return false;
        value=data_queue.front();
        data_queue.pop();
        return true;
    }

    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lk(mut);
        if(data_queue.empty())
            return std::shared_ptr<T>();
        std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
        data_queue.pop();
        return res;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }
};

// Lazy instantiation of snarkvm_t
class snarkvm_singleton_t {
    bool failed = false;
    int  iindex  = 0;
    snarkvm_t *snarkvm = nullptr;

public:
    snarkvm_singleton_t(int ii) {
        iindex = ii;
    }
    ~snarkvm_singleton_t() {
        delete snarkvm;
        snarkvm = nullptr;
    }
    bool ok() {
        if (!failed && snarkvm == nullptr) {
            // SNP TODO: max domain size?
            snarkvm = new snarkvm_t(17);
            if (snarkvm == nullptr) {
                failed = true;
            }
        }
        cout << "-----------------------vm ok----------------------";

        return snarkvm != nullptr;
    }
    snarkvm_t* get() {
        assert (ok());
        return snarkvm;
    }
    int get_index(){
        return iindex;
    }
};
//snarkvm_singleton_t snarkvm_g;

static threadsafe_queue<snarkvm_singleton_t*> snarkvm_g;
bool initCode()
{
    for (int i = 0; i < 32 /*!!*/; i++) {
        snarkvm_g.push(new snarkvm_singleton_t(i));
    }
    return true;
}

static bool bSzArrCountryCodeInit  = initCode();


#ifndef __CUDA_ARCH__

extern "C" {
RustError snarkvm_ntt(fr_t* inout, uint32_t lg_domain_size,
                      NTT::InputOutputOrder ntt_order, NTT::Direction ntt_direction,
                      NTT::Type ntt_type)
{

    std::shared_ptr<snarkvm_singleton_t*> p = snarkvm_g.wait_and_pop();

    RustError ret = RustError{cudaErrorMemoryAllocation};
    try{
        if ((*p)->ok()) {
            ret = (*p)->get()->NTT(inout, inout, lg_domain_size, ntt_order,
                                   ntt_direction, ntt_type);
            snarkvm_g.push((*p));
            return ret;
        }
        snarkvm_g.push((*p));
    }
    catch (...){
        snarkvm_g.push((*p));
    }
    return ret;

    //if (!snarkvm_g.ok()) {
    //    return RustError{cudaErrorMemoryAllocation};
    //}
    //return snarkvm_g->NTT(inout, inout, lg_domain_size, ntt_order,
    //                      ntt_direction, ntt_type);
}

RustError snarkvm_polymul(fr_t* out,
                          size_t pcount, fr_t** polynomials, size_t* plens,
                          size_t ecount, fr_t** evaluations, size_t* elens,
                          uint32_t lg_domain_size) {

    std::shared_ptr<snarkvm_singleton_t*> p = snarkvm_g.wait_and_pop();

    RustError ret = RustError{cudaErrorMemoryAllocation};
    try{
        if ((*p)->ok()) {

            ret = (*p)->get()->PolyMul(out,
                                       pcount, polynomials, plens,
                                       ecount, evaluations, elens,
                                       lg_domain_size);
            snarkvm_g.push((*p));
            return ret;
        }
        snarkvm_g.push((*p));
    }
    catch (...){
        snarkvm_g.push((*p));
    }
    return ret;


    //if (!snarkvm_g.ok()) {
    //    return RustError{cudaErrorMemoryAllocation};
    //}
    //return snarkvm_g->PolyMul(out,
    //                          pcount, polynomials, plens,
    //                          ecount, evaluations, elens,
    //                          lg_domain_size);
}

RustError snarkvm_msm(point_t* out, const affine_t points[], size_t npoints,
                      const scalar_t scalars[], size_t ffi_affine_size) {

    high_resolution_clock::time_point beginTime = high_resolution_clock::now();

    std::shared_ptr<snarkvm_singleton_t*> p = snarkvm_g.wait_and_pop();

    high_resolution_clock::time_point endTime = high_resolution_clock::now();

    milliseconds timeInterval = std::chrono::duration_cast<milliseconds>(endTime - beginTime);
    cout <<  "### snarkvm_msm wait_and_pop time " << timeInterval.count() << "ms\n";

    RustError ret = RustError{cudaErrorMemoryAllocation};
    try{
        if ((*p)->ok()) {
            cout << "vm index: " << (*p)->get_index()  << "\r\n";
            ret = (*p)->get()->MSM(out, points, npoints, scalars, ffi_affine_size);
            snarkvm_g.push((*p));
            return ret;
        }
        snarkvm_g.push((*p));
    }
    catch (...){
        snarkvm_g.push((*p));
    }
    return ret;

    //if (!snarkvm_g.ok()) {
    //    return RustError{cudaErrorMemoryAllocation};
    //}
    //return snarkvm_g->MSM(out, points, npoints, scalars, ffi_affine_size);
}
}
#endif // __CUDA_ARCH__

#endif
