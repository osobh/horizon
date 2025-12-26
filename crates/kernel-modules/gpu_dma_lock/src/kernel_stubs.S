/*
 * Assembly stubs for Rust standard library functions
 * These are required for kernel module linking
 */

.text

/* Rust core::fmt functions for debugging */
.global _ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u64$GT$3fmt17hc038dffd7e2f7fd6E
_ZN4core3fmt3num3imp52_$LT$impl$u20$core..fmt..Display$u20$for$u20$u64$GT$3fmt17hc038dffd7e2f7fd6E:
    ret

.global _ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17hb9928f289d6fcf72E
_ZN4core3fmt3num3imp54_$LT$impl$u20$core..fmt..Display$u20$for$u20$usize$GT$3fmt17hb9928f289d6fcf72E:
    ret

.global _ZN42_$LT$str$u20$as$u20$core..fmt..Display$GT$3fmt17hc34a6706652b50c1E
_ZN42_$LT$str$u20$as$u20$core..fmt..Display$GT$3fmt17hc34a6706652b50c1E:
    ret

/* Rust allocation error handling */
.global _ZN5alloc5alloc18handle_alloc_error17h29c279d8237d34e5E
_ZN5alloc5alloc18handle_alloc_error17h29c279d8237d34e5E:
    call panic

.global _ZN5alloc7raw_vec12handle_error17h891236e332f51b87E
_ZN5alloc7raw_vec12handle_error17h891236e332f51b87E:
    call panic

/* Rust OOM handler */
.global _RNvCs691rhTbG0Ee_7___rustc9___rdl_oom
_RNvCs691rhTbG0Ee_7___rustc9___rdl_oom:
    call panic

/* Rust format functions */
.global _ZN5alloc3fmt6format12format_inner17h8a62f782d788077eE
_ZN5alloc3fmt6format12format_inner17h8a62f782d788077eE:
    ret

/* Rust slice index error handlers */
.global _ZN4core5slice5index24slice_end_index_len_fail17hb6890d29d4255062E
_ZN4core5slice5index24slice_end_index_len_fail17hb6890d29d4255062E:
    call panic

.global _ZN4core5slice5index26slice_start_index_len_fail17h0596e605fb4610d0E
_ZN4core5slice5index26slice_start_index_len_fail17h0596e605fb4610d0E:
    call panic

/* UTF-8 conversion */
.global _ZN4core3str8converts9from_utf817h643f4cdf3dbea228E
_ZN4core3str8converts9from_utf817h643f4cdf3dbea228E:
    ret

/* Additional core panic functions */
.global _ZN4core9panicking9panic_fmt17hc8737e8cca20a7c8E
_ZN4core9panicking9panic_fmt17hc8737e8cca20a7c8E:
    call panic

.global _ZN4core6option13unwrap_failed17h62317944fa5dc382E
_ZN4core6option13unwrap_failed17h62317944fa5dc382E:
    call panic

.global _ZN4core9panicking5panic17h4a11c031239f36a8E
_ZN4core9panicking5panic17h4a11c031239f36a8E:
    call panic

.global _ZN4core5slice4sort6shared9smallsort22panic_on_ord_violation17h5323cfd9f2ada983E
_ZN4core5slice4sort6shared9smallsort22panic_on_ord_violation17h5323cfd9f2ada983E:
    call panic

.global _ZN4core6option13expect_failed17h1fcc4e32848a6083E
_ZN4core6option13expect_failed17h1fcc4e32848a6083E:
    call panic

/* Math functions */
.global _ZN4core5slice4sort6stable5drift11sqrt_approx17hafe01fb2b90a9983E
_ZN4core5slice4sort6stable5drift11sqrt_approx17hafe01fb2b90a9983E:
    /* Simple sqrt approximation for kernel */
    mov %rdi, %rax
    ret

/* Memory functions to avoid GOTPCREL */
.global memcpy
memcpy:
    /* Simple memcpy implementation */
    mov %rdx, %rcx
    mov %rdi, %rax
    rep movsb
    ret

.global memset
memset:
    /* Simple memset implementation */
    mov %rdx, %rcx
    mov %sil, %al
    mov %rdi, %r8
    rep stosb
    mov %r8, %rax
    ret