#![allow(dead_code)]

use std::alloc::{self, Layout};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeFull, RangeFrom, RangeTo};
use std::ptr::NonNull;

pub const ALIGN_AVX512: usize = 64;
pub const ALIGN_AVX2: usize = 32;
pub const ALIGN_SSE: usize = 16;

#[derive(Clone, Copy)]
pub struct AlignedAlloc<const ALN: usize = ALIGN_AVX512>;

pub type AlignedVec64<T> = AlignedVec<T, ALIGN_AVX512>;
pub type AlignedVec32<T> = AlignedVec<T, ALIGN_AVX2>;

pub struct AlignedVec<T, const ALN: usize = ALIGN_AVX512> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
    _marker: PhantomData<T>,
}

impl<T, const ALN: usize> AlignedVec<T, ALN> {
    pub fn new() -> Self {
        assert!(ALN.is_power_of_two(), "ALN must be a power of two");
        assert!(ALN >= std::mem::align_of::<T>(), "ALN must be >= align_of::<T>()");
        AlignedVec {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut v = Self::new();
        if capacity > 0 {
            v.grow(capacity);
        }
        v
    }

    pub fn zeros(len: usize) -> Self {
        let mut v = Self::with_capacity(len);
        unsafe {
            std::ptr::write_bytes(v.ptr.as_ptr(), 0, len);
            v.len = len;
        }
        v
    }

    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Copy,
    {
        let mut v = Self::with_capacity(slice.len());
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), v.ptr.as_ptr(), slice.len());
            v.len = slice.len();
        }
        v
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
            self.grow(new_cap);
        }
        unsafe {
            std::ptr::write(self.ptr.as_ptr().add(self.len), value);
            self.len += 1;
        }
    }

    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.cap {
            self.grow(new_len);
        }
        if new_len > self.len {
            unsafe {
                let start = self.ptr.as_ptr().add(self.len);
                for i in 0..new_len - self.len {
                    std::ptr::write(start.add(i), value.clone());
                }
            }
        } else if new_len < self.len {
            unsafe {
                for i in new_len..self.len {
                    std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
            }
        }
        self.len = new_len;
    }

    pub fn clear(&mut self) {
        unsafe {
            for i in 0..self.len {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
            self.len = 0;
        }
    }

    fn grow(&mut self, new_cap: usize) {
        assert!(new_cap > self.cap);

        let size = std::mem::size_of::<T>();
        let old_layout = if self.cap > 0 {
            Some(Layout::from_size_align(self.cap * size, ALN).unwrap())
        } else {
            None
        };
        let new_layout = Layout::from_size_align(new_cap * size, ALN).unwrap();

        let new_ptr = unsafe {
            if let Some(old_lay) = old_layout {
                let old_raw = self.ptr.as_ptr() as *mut u8;
                let new_raw = alloc::realloc(old_raw, old_lay, new_layout.size());
                if new_raw.is_null() {
                    alloc::handle_alloc_error(new_layout);
                }
                new_raw as *mut T
            } else {
                let new_raw = alloc::alloc(new_layout);
                if new_raw.is_null() {
                    alloc::handle_alloc_error(new_layout);
                }
                new_raw as *mut T
            }
        };

        self.ptr = NonNull::new(new_ptr).expect("non-null");
        self.cap = new_cap;
    }

    pub fn alignment() -> usize {
        ALN
    }

    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % ALN == 0
    }
}

impl<T, const ALN: usize> Default for AlignedVec<T, ALN> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const ALN: usize> Drop for AlignedVec<T, ALN> {
    fn drop(&mut self) {
        if self.cap > 0 {
            unsafe {
                for i in 0..self.len {
                    std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
                let size = std::mem::size_of::<T>() * self.cap;
                let layout = Layout::from_size_align(size, ALN).unwrap();
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T, const ALN: usize> Deref for AlignedVec<T, ALN> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T, const ALN: usize> DerefMut for AlignedVec<T, ALN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Clone, const ALN: usize> Clone for AlignedVec<T, ALN> {
    fn clone(&self) -> Self {
        let mut v = Self::with_capacity(self.len);
        unsafe {
            for i in 0..self.len {
                std::ptr::write(v.ptr.as_ptr().add(i), (*self.ptr.as_ptr().add(i)).clone());
            }
            v.len = self.len;
        }
        v
    }
}

impl<T: fmt::Debug, const ALN: usize> fmt::Debug for AlignedVec<T, ALN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedVec")
            .field("len", &self.len)
            .field("cap", &self.cap)
            .field("align", &ALN)
            .field("ptr_aligned", &self.is_aligned())
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T: PartialEq, const ALN: usize> PartialEq for AlignedVec<T, ALN> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T, const ALN: usize> Index<usize> for AlignedVec<T, ALN> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        assert!(idx < self.len, "index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(idx) }
    }
}

impl<T, const ALN: usize> IndexMut<usize> for AlignedVec<T, ALN> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        assert!(idx < self.len, "index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(idx) }
    }
}

impl<T, const ALN: usize> Index<Range<usize>> for AlignedVec<T, ALN> {
    type Output = [T];
    fn index(&self, range: Range<usize>) -> &Self::Output {
        assert!(range.start <= range.end && range.end <= self.len, "range out of bounds");
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(range.start), range.end - range.start) }
    }
}

impl<T, const ALN: usize> IndexMut<Range<usize>> for AlignedVec<T, ALN> {
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        assert!(range.start <= range.end && range.end <= self.len, "range out of bounds");
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(range.start), range.end - range.start) }
    }
}

impl<T, const ALN: usize> Index<RangeFull> for AlignedVec<T, ALN> {
    type Output = [T];
    fn index(&self, _: RangeFull) -> &Self::Output {
        self.as_slice()
    }
}

impl<T, const ALN: usize> IndexMut<RangeFull> for AlignedVec<T, ALN> {
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        self.as_mut_slice()
    }
}

impl<T, const ALN: usize> Index<RangeFrom<usize>> for AlignedVec<T, ALN> {
    type Output = [T];
    fn index(&self, range: RangeFrom<usize>) -> &Self::Output {
        &self.as_slice()[range.start..]
    }
}

impl<T, const ALN: usize> IndexMut<RangeFrom<usize>> for AlignedVec<T, ALN> {
    fn index_mut(&mut self, range: RangeFrom<usize>) -> &mut Self::Output {
        &mut self.as_mut_slice()[range.start..]
    }
}

impl<T, const ALN: usize> Index<RangeTo<usize>> for AlignedVec<T, ALN> {
    type Output = [T];
    fn index(&self, range: RangeTo<usize>) -> &Self::Output {
        &self.as_slice()[..range.end]
    }
}

impl<T, const ALN: usize> IndexMut<RangeTo<usize>> for AlignedVec<T, ALN> {
    fn index_mut(&mut self, range: RangeTo<usize>) -> &mut Self::Output {
        &mut self.as_mut_slice()[..range.end]
    }
}

impl<T: Copy, const ALN: usize> From<&[T]> for AlignedVec<T, ALN> {
    fn from(slice: &[T]) -> Self {
        Self::from_slice(slice)
    }
}

unsafe impl<T: Send, const ALN: usize> Send for AlignedVec<T, ALN> {}
unsafe impl<T: Sync, const ALN: usize> Sync for AlignedVec<T, ALN> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_alignment_64() {
        let v: AlignedVec<f32> = AlignedVec::zeros(1024);
        let addr = v.as_ptr() as usize;
        assert_eq!(addr % 64, 0, "ptr {:p} not 64-byte aligned", v.as_ptr());
        assert!(v.is_aligned());
    }

    #[test]
    fn test_alignment_32() {
        let v: AlignedVec32<f64> = AlignedVec::zeros(1024);
        let addr = v.as_ptr() as usize;
        assert_eq!(addr % 32, 0, "ptr {:p} not 32-byte aligned", v.as_ptr());
    }

    #[test]
    fn test_push_and_get() {
        let mut v: AlignedVec<i32> = AlignedVec::new();
        for i in 0..1000 {
            v.push(i);
        }
        assert_eq!(v.len(), 1000);
        assert_eq!(v[0], 0);
        assert_eq!(v[999], 999);
    }

    #[test]
    fn test_from_slice() {
        let data: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let v = AlignedVec::<f32>::from_slice(&data);
        assert_eq!(v.len(), 256);
        assert_eq!(&v[..], &data[..]);
        assert!(v.is_aligned());
    }

    #[test]
    fn test_clone() {
        let mut v: AlignedVec<String> = AlignedVec::new();
        v.push("hello".to_string());
        v.push("world".to_string());
        let v2 = v.clone();
        assert_eq!(v, v2);
        assert!(v.is_aligned());
        assert!(v2.is_aligned());
    }

    #[test]
    fn test_complex64_alignment() {
        use num_complex::Complex64 as C64;
        let v: AlignedVec<C64> = AlignedVec::zeros(4096);
        let addr = v.as_ptr() as usize;
        assert_eq!(addr % 64, 0, "Complex64 ptr not 64-byte aligned");
        assert_eq!(mem::size_of::<C64>(), 16);
    }
}
