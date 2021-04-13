#ifndef IMG_UTILS_H
#define IMG_UTILS_H

float * read_image_asfloat(char *imgName,int *nC, int *nR, int *colors) ;
void write_image_fromfloat(char *filename,float *img, int nC, int nR, int colors);

#endif /* IMG_UTILS_H */
