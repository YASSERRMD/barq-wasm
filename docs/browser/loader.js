// Bundle loader with real SIMD feature detection.
//
// Detects wasm SIMD support by validating a minimal module that uses a v128
// instruction (the same approach as the wasm-feature-detect library), then
// loads the matching bundle. The simd bundle is NEVER loaded on browsers
// without SIMD support.

// A minimal module: (func (result v128) i32.const 0 i8x16.splat i8x16.popcnt)
const SIMD_DETECT_MODULE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1,
  8, 0, 65, 0, 253, 15, 253, 98, 11,
]);

export function simdSupported() {
  return WebAssembly.validate(SIMD_DETECT_MODULE);
}

/**
 * Load the best bundle for this browser.
 * @param {string} base - base URL containing pkg/scalar and pkg/simd
 * @returns {{module: object, bundle: "simd"|"scalar"}}
 */
export async function loadBarqWasm(base = ".") {
  const bundle = simdSupported() ? "simd" : "scalar";
  const mod = await import(`${base}/pkg/${bundle}/barq_wasm.js`);
  await mod.default();
  // Cross-check: the bundle must agree about how it was compiled.
  const compiledWithSimd = mod.simd128_enabled();
  if (compiledWithSimd !== (bundle === "simd")) {
    throw new Error(
      `bundle mismatch: loaded '${bundle}' but simd128_enabled()=${compiledWithSimd}`,
    );
  }
  return { module: mod, bundle };
}
