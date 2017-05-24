#define UBYTE_IMAGE_MAGIC 2051
#define UBYTE_LABEL_MAGIC 2049

#define bswap(x) __builtin_bswap32(x)

#pragma pack(push, 1)
struct UByteImageDataset {
  uint32_t magic;
  uint32_t length;
  uint32_t height;
  uint32_t width;
  void Swap() {
    magic = bswap(magic);
    length = bswap(length);
    height = bswap(height);
    width = bswap(width);
  }
};

struct UByteLabelDataset {
  uint32_t magic;
  uint32_t length;
  void Swap() {
    magic = bswap(magic);
    length = bswap(length);
  }
};
#pragma pack(pop)

size_t ReadUByteDataset(const char *image_filename, const char *label_filename,
                        uint8_t *data, uint8_t *labels, size_t& width, size_t& height) {
    UByteImageDataset image_header;
    UByteLabelDataset label_header;
    FILE * imfp = fopen(image_filename, "r");
    FILE * lbfp = fopen(label_filename, "r");
    fread(&image_header, sizeof(UByteImageDataset), 1, imfp);
    fread(&label_header, sizeof(UByteLabelDataset), 1, lbfp);
    image_header.Swap();
    label_header.Swap();
    width = image_header.width;
    height = image_header.height;
    printf("%lu %lu %u\n",width,height,image_header.length);
    if(data != NULL)
      fread(data, sizeof(uint8_t), image_header.length * width * height, imfp);
    if(labels != NULL)
      fread(labels, sizeof(uint8_t), label_header.length, lbfp);
    fclose(imfp);
    fclose(lbfp);

    return image_header.length;
}
