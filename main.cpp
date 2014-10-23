#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int main(int argc, char *argv[])
{
    Mat im_src, im_segmented, im_grey, im_normalized;

    /// Load an image
    im_src = imread( argv[1] );

    if( !im_src.data )
    { return -1; }

    /// Create window
    namedWindow( "src", CV_WINDOW_AUTOSIZE );
    namedWindow( "segmented", CV_WINDOW_AUTOSIZE );

    cvtColor(im_src, im_grey, CV_BGR2GRAY);
    cvtColor(im_src, im_segmented, CV_BGR2GRAY);
    cvtColor(im_src, im_normalized, CV_BGR2GRAY);

    //--------------Image Segmentation
    int segmentation_block_size = 16;
    for (int i = 0; i < im_grey.cols; i = i + segmentation_block_size)
    {
        for (int j = 0; j < im_grey.rows; j = j + segmentation_block_size)
        {
            double mean_val = 0;
            for (int _i = 0; _i < segmentation_block_size; _i++)
            {
                for (int _j = 0; _j < segmentation_block_size; _j++)
                {
                    mean_val = mean_val + im_grey.at<uchar>(Point(i + _i, j + _j));
                }
            }
            mean_val = mean_val/pow(segmentation_block_size, 2);

            double sub_var = 0;
            for (int _i = 0; _i < segmentation_block_size; _i++)
            {
                for (int _j = 0; _j < segmentation_block_size; _j++)
                {
                    sub_var = sub_var + pow((im_grey.at<uchar>(Point(i + _i, j + _j)) - mean_val), 2.0);
                }
            }
            sub_var = sub_var / pow(segmentation_block_size, 2.0);

            for (int _i = 0; _i < segmentation_block_size; _i++)
            {
                for (int _j = 0; _j < segmentation_block_size; _j++)
                {
                    if (sub_var > 100) im_segmented.at<uchar>(Point(i+_i,j+_j)) = im_grey.at<uchar>(Point(i+_i,j+_j));
                    else im_segmented.at<uchar>(Point(i+_i,j+_j)) = 0;
                }
            }
        }
    }

    //--------------Image Normalization
    double mean = 0, variance = 0;
    double desired_mean = 0, desired_variance = 1;

    for (int i = 0; i < im_grey.rows; i++)
    {
        for (int j = 0; j < im_grey.cols; j++)
        {
            mean = mean + im_segmented.at<uchar>(Point(i, j));
        }
    }
    mean = mean / (im_segmented.rows*im_segmented.cols);

    for (int i = 0; i < im_grey.rows; i++)
    {
        for (int j = 0; j < im_grey.cols; j++)
        {
            variance = variance + pow(im_segmented.at<uchar>(Point(i, j)) - mean, 2);
        }
    }
    mean = mean / (im_segmented.rows*im_segmented.cols);

    for (int j = 0; j < im_grey.rows; j++)
    {
        for (int i = 0; i < im_grey.cols; i++)
        {
            if (im_segmented.at<uchar>(Point(i, j)) > mean) im_normalized.at<uchar>(Point(i, j)) = desired_mean + pow((desired_variance*pow(im_segmented.at<uchar>(Point(i, j)) - mean, 2))/variance, 0.5);
            else im_normalized.at<uchar>(Point(i, j)) = desired_mean - pow((desired_variance*pow(im_segmented.at<uchar>(Point(i, j)) - mean, 2))/variance, 0.5);
        }
    }

    imshow( "src", im_segmented );
    imshow( "segmented", im_normalized );

    waitKey(0);
    return 0;
}
