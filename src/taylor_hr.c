void talyer_espcn_cpu(float* data_im,
     int channels,  int scale, int height,  int width,
     int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int i, j, k;
    int dx[4] = {1, -1, 1, -1};
    int dy[4] = {1, 1, -1, -1};

    int height_col = height + 2*pad;
    int width_col = width + 2*pad;


    
    for (h = 0; h < height; ++h){
        for (w = 0; w < width; ++w){
            for (i = 0; i < 4; ++i){
                for (j = 0; j < 3; j++){
                    for (k = 0; k < 3; k++){
                        int col_index = (scale*(h + pad)) * (scale*width) + (scale*w) + (dx[i]*k) +(dy[i]*k*scale*width);
                        data_col[col_index] = data_im[(c * height + h) * width + w];            

                    }

                }

            }
        }
    }
}
