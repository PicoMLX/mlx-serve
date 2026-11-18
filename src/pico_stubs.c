// What libPicoMlxServe.dylib (build/pico.zig) leaves out, answered as unavailable: the tree's media, mesh and chat
// template code compiles into the one unit but nothing in the library's interface reaches it (prompts are token ids,
// replies are token ids), so libwebp, stb_image, xatlas, FQMS and the Jinja renderer are not linked. src/ane_stub.c
// does the same for the Neural Engine bridge.
#include <stddef.h>
#include <stdint.h>

// libwebp: undecodable.
uint8_t *WebPDecodeRGBA(const uint8_t *data, size_t size, int *w, int *h) { (void)data; (void)size; (void)w; (void)h; return NULL; }
uint8_t *WebPDecodeRGB(const uint8_t *data, size_t size, int *w, int *h) { (void)data; (void)size; (void)w; (void)h; return NULL; }
void WebPFree(void *ptr) { (void)ptr; }

// stb_image, stb_image_write: undecodable, nothing written.
unsigned char *stbi_load_from_memory(const unsigned char *buffer, int len, int *x, int *y, int *channels, int desired) {
    (void)buffer; (void)len; (void)x; (void)y; (void)channels; (void)desired;
    return NULL;
}
int stbi_info_from_memory(const unsigned char *buffer, int len, int *x, int *y, int *comp) {
    (void)buffer; (void)len; (void)x; (void)y; (void)comp;
    return 0;
}
void stbi_image_free(void *p) { (void)p; }
int stbi_write_png_to_func(void *func, void *context, int w, int h, int comp, const void *data, int stride) {
    (void)func; (void)context; (void)w; (void)h; (void)comp; (void)data; (void)stride;
    return 0;
}
int stbi_write_jpg_to_func(void *func, void *context, int x, int y, int comp, const void *data, int quality) {
    (void)func; (void)context; (void)x; (void)y; (void)comp; (void)data; (void)quality;
    return 0;
}

// FQMS decimation (lib/fqms/fqms_shim.h): no result.
void *fqms_simplify(const float *positions, uint32_t vertex_count, const uint32_t *indices, uint32_t index_count, uint32_t target_faces) {
    (void)positions; (void)vertex_count; (void)indices; (void)index_count; (void)target_faces;
    return NULL;
}
uint32_t fqms_vertex_count(const void *r) { (void)r; return 0; }
uint32_t fqms_index_count(const void *r) { (void)r; return 0; }
const float *fqms_positions(const void *r) { (void)r; return NULL; }
const uint32_t *fqms_indices(const void *r) { (void)r; return NULL; }
void fqms_free(void *r) { (void)r; }

// xatlas UV unwrap (lib/xatlas/xatlas_shim.h): no result.
void *uvw_parametrize(const float *positions, uint32_t vertex_count, const uint32_t *indices, uint32_t index_count, const uint32_t *face_material) {
    (void)positions; (void)vertex_count; (void)indices; (void)index_count; (void)face_material;
    return NULL;
}
uint32_t uvw_vertex_count(const void *r) { (void)r; return 0; }
uint32_t uvw_index_count(const void *r) { (void)r; return 0; }
uint32_t uvw_chart_count(const void *r) { (void)r; return 0; }
uint32_t uvw_atlas_width(const void *r) { (void)r; return 0; }
uint32_t uvw_atlas_height(const void *r) { (void)r; return 0; }
const uint32_t *uvw_vmapping(const void *r) { (void)r; return NULL; }
const uint32_t *uvw_indices(const void *r) { (void)r; return NULL; }
const float *uvw_uvs(const void *r) { (void)r; return NULL; }
void uvw_free(void *r) { (void)r; }

// The Jinja renderer (lib/jinja_cpp/jinja_wrapper.h): every render fails.
char *jinja_render_chat(const char *template_str, const char *messages_json, const char *tools_json, const char *extra_json, int add_generation_prompt, size_t *out_len) {
    (void)template_str; (void)messages_json; (void)tools_json; (void)extra_json; (void)add_generation_prompt;
    if (out_len != NULL) *out_len = 0;
    return NULL;
}
void jinja_str_free(char *s) { (void)s; }
const char *jinja_last_error(void) { return "chat templates are not rendered by this library"; }
