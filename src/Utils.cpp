#include "Utils.h"

double round(double d)
{
	return floor(d + 0.5);
}

double sigmoid(double z)
{
	return 1.0 / (1.0 + exp(-z));
}

double sigmoidGradient(double z)
{
	double gz = sigmoid(z);

	return gz * (1 - gz);
}

string vformat(const char *fmt, va_list ap)
{
    // Allocate a buffer on the stack that's big enough for us almost
    // all the time.  Be prepared to allocate dynamically if it doesn't fit.
    size_t size = 1024;
    char stackbuf[1024];
    std::vector<char> dynamicbuf;
    char *buf = &stackbuf[0];

    while (1) 
	{
        // Try to vsnprintf into our buffer.
        int needed = vsnprintf (buf, size, fmt, ap);
        // NB. C99 (which modern Linux and OS X follow) says vsnprintf
        // failure returns the length it would have needed.  But older
        // glibc and current Windows return -1 for failure, i.e., not
        // telling us how much was needed.

        if (needed <= (int)size && needed >= 0) 
		{
            // It fit fine so we're done.
            return std::string (buf, (size_t) needed);
        }

        // vsnprintf reported that it wanted to write more characters
        // than we allotted.  So try again using a dynamic buffer.  This
        // doesn't happen very often if we chose our initial size well.
        size = (needed > 0) ? (needed+1) : (size*2);
        dynamicbuf.resize (size);
        buf = &dynamicbuf[0];
    }
}

string formatString(const char *fmt, ...)
{
    va_list ap;
    va_start (ap, fmt);
    std::string buf = vformat (fmt, ap);
    va_end (ap);
    return buf;
}

int reverseInt(int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist(const char* Xfile, MatrixXd& X, const char* yfile, VectorXi& y, unsigned int maxImages)
{
	ifstream images(Xfile, ios::binary);
	ifstream labels(yfile, ios::binary);

	if (images.is_open() && labels.is_open())
	{
		int imagesMagicNumber = 0;
		int imagesCount = 0;
		int imagesRows = 0;
		int imagesCols = 0;

		images.read((char*)&imagesMagicNumber, sizeof(imagesMagicNumber));
		imagesMagicNumber = reverseInt(imagesMagicNumber);

		images.read((char*)&imagesCount, sizeof(imagesCount));
		imagesCount = reverseInt(imagesCount);

		images.read((char*)&imagesRows, sizeof(imagesRows));
		imagesRows = reverseInt(imagesRows);

		images.read((char*)&imagesCols,sizeof(imagesCols));
		imagesCols = reverseInt(imagesCols);

		int labelsMagicNumber = 0;
		int labelsCount = 0;
		
		labels.read((char*)&labelsMagicNumber, sizeof(labelsMagicNumber));
		labelsMagicNumber = reverseInt(labelsMagicNumber);

		labels.read((char*)&labelsCount, sizeof(labelsCount));
		labelsCount = reverseInt(labelsCount);
		
		if (imagesCount != labelsCount)
		{
			throw "Images and labels counts don't match";
		}

		if (maxImages != 0 && imagesCount > maxImages)
		{
			imagesCount = maxImages;
		}
		
		X.resize(imagesCount, imagesRows * imagesCols);
		y.resize(imagesCount);

		unsigned char temp = 0;

		for(int i = 0; i < imagesCount; i++)
		{
			for(int r = 0; r < imagesRows; r++)
			{
				for(int c = 0; c < imagesCols; c++)
				{
					images.read((char*)&temp, sizeof(temp));
					X(i, (r * imagesCols) + c) = 1.0 - ((double)temp) / 255.0;
					//X(i, (r * imagesCols) + c) = (double)temp / 255.0;
					//X(i, (r * imagesCols) + c) = 1;
				}
			}
			
			labels.read((char*)&temp, sizeof(temp));
			y(i) = (int)temp;			
		}

		images.close();
		labels.close();
	}
}

void displayData(MatrixXd X)
{
	unsigned int example_width = round(sqrt((double)X.cols()));
	unsigned int example_height = (X.cols() / example_width);	
		
	unsigned int display_rows = floor(sqrt((double)X.rows()));
	unsigned int display_cols = ceil((double)X.rows() / display_rows);

	unsigned int pad = 1;

	
	Mat displayArray = -Mat::ones(pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad), CV_64FC1);

	unsigned int curr_ex = 0;

	for (unsigned int j = 0; j < display_rows; j++)
	{
		for (unsigned int i = 0; i < display_cols; i++)
		{
			if (curr_ex >= X.rows())
			{
				break; 
			}
						
			double max_val = X.row(curr_ex).cwiseAbs().maxCoeff();
			
			for (unsigned int r = 0; r < 28; r++)
			{
				for (unsigned int c = 0; c < 28; c++)
				{
					displayArray.at<double>(pad + j * (example_height + pad) + r,  pad + i * (example_width + pad) + c) = X.row(curr_ex)((r * 28) + c) / max_val;
				}
			}			

			curr_ex++;
		}

		if (curr_ex > X.rows())
		{
			break; 
		}
	}

	Mat resized(displayArray.rows * 4, displayArray.cols * 4, displayArray.type());
		
	resize(displayArray, resized, resized.size());

	imshow("Hidden Layer", resized);

	waitKey(10);
}