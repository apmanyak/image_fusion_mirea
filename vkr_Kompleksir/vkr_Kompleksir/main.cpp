
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

const int histSize = 256;

//�������
//�������� �����������
void printWindow(string word, Mat a)
{
	namedWindow(word, WINDOW_AUTOSIZE);
	imshow(word, a);
}

// _______���������� ����������� ������������ �����������_______
Mat cHist(Mat &seq)
{

	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	/// Compute the histograms:
	calcHist(&seq, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	return hist;
}
//_______���������� ����������� ����������� �����������________
void paintHistGrey(Mat &hist, string word)
{

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));

	//����������� ���������� 
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage,
			Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	printWindow(word, histImage);
}


/*___________________������� ���������� � ���������� ����������� RGB �����������____________________*/

void calchistRGB(const Mat &Image, Mat &r_hist, Mat &g_hist, Mat &b_hist, string word)
{
	float range[] = { 0,256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	vector<Mat> bgr_planes;
	split(Image, bgr_planes);

	//���������� �����������
	calcHist(&bgr_planes[0], 1, 0, noArray(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, noArray(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, noArray(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    //���������� �����������

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	//����������� ���������� 
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//��������� �������� ��� ������ ������ 
	for (int i = 1; i < histSize; i++)
	{
		line(histImage,
			Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage,
			Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage,
			Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}
	printWindow(word, histImage);
}


// ������������� ��������
float entropy(Mat &seq, Size size)
{
	Mat hist = cHist(seq);
	int cnt = 0;
	float entr = 0;
	int MN = size.width * size.height; //total size of all symbols in an image

	for (int i = 0; i<histSize; i++)
	{
		float X = hist.at<float>(i, 0); //the number of times a sybmol has occured
		if (X>0) //log of zero goes to infinity
		{
			cnt++;
			float p = X / MN;
			entr += (p)*(log2(1 / p));
		}
	}


	return entr;

}

//�������� ���� ������� �����������
vector<Mat> showSeparatedChannels(vector<Mat> channels) {
	vector<Mat> separatedChannels; //create each image for each channel 
	for ( int i = 0 ; i < 3 ; i++){ 
		Mat zer=Mat::zeros( channels[0].rows, channels[0].cols, channels[0].type()); 
		vector<Mat> aux; 
		for (int j=0; j < 3 ; j++){
			if(j==i)
				aux.push_back(channels[i]); 
			else
				aux.push_back(zer);
		} 
		Mat chann; 
		merge(aux, chann); 
		separatedChannels.push_back(chann);
	} 
	return separatedChannels;
}

//������
 int main()
{
	 setlocale(LC_ALL, "Rus");
	Mat WEB = imread("rgb_stol_Affin.jpg");//rgb_stol_Affin.jpg
	Mat IR = imread("stol_IR.jpg",0); //stol_IR.jpg
	printWindow("WEB",WEB);
	printWindow("IR", IR);

	Mat WEBgr;
	cvtColor(WEB, WEBgr, COLOR_BGR2GRAY);//WEB-����������� �����������
	printWindow("WEBgr", WEBgr);
	imwrite("WEBgr.jpg", WEBgr);
	
	Mat IR_f;

//����� �������
	//////������ ������ �� �� �����������
	//GaussianBlur(IR, IR_f, Size(7, 7), 0, 0);
	//printWindow("�������� ���������", IR_f);
	//imwrite("IRgauss.jpg", IR_f);

	//��������� ���������� �� �����������
	medianBlur(IR, IR_f, 9);
	printWindow("��������� ������", IR_f);
	imwrite("IRmedian.jpg", IR_f);

	//////������������� �������� �� �� �����������
	//bilateralFilter(IR, IR_f, -1, 9, 4);
	//printWindow("������������ ���������", IR_f);
	//imwrite("2xstoron_filter.jpg", IR_f);

	//���������� ����������� ��-�����������
	Mat hist_IR = cHist(IR_f);
	paintHistGrey(hist_IR, "����������� �� �����������");

	//���������� ����������� WEB-����������� ����������� 
	Mat hist_WEBgr = cHist(WEBgr);
	//paintHistGrey(hist_WEBgr, "����������� WEB-������������ �����������");

	vector<Mat> channels;
	split(WEB,channels);
	//������ � ��������� ������
	//printWindow("B(grey)", channels[0]);//������ �����
	//printWindow("G(grey)", channels[1]);//������ �����
	//printWindow("R(grey)", channels[2]);//������ �����
	//imwrite("B(grey).jpg", channels[0]);
	//imwrite("G(grey).jpg", channels[1]);
	//imwrite("R(grey).jpg", channels[2]);
	Mat R, G, B;
	channels[2].copyTo(R);
	channels[1].copyTo(G);
	channels[0].copyTo(B);

	cout << "��������, ����� B :" << entropy(channels[0], channels[0].size()) << endl;
	cout << "��������, ����� G :" << entropy(channels[1], channels[1].size()) << endl;
	cout << "��������, ����� R :" <<  entropy(channels[2], channels[2].size()) << endl;
	cout << "�������� ������������ �� �����������: "<< entropy(IR_f,IR_f.size()) << endl;
	cout << "�������� WEB-������������ �����������: " << entropy(WEBgr, WEBgr.size()) << endl;


	Mat a = IR_f;
	Mat b = WEBgr;
	const int MN = WEB.cols*WEB.rows;

	//
	// ���������������� �� ������ ����������� ������������
	//���������������� ��+���
	Mat z4;
	z4 = b / 2 + a / 2;
	imwrite("kompleksir_WEB_IR.jpg", z4);
	cout << "�������� ����������� Z4: " << entropy(z4, z4.size()) << endl;

	//���������������� ��+B-�����
	Mat z1;
	z1= B/2  + a/2;
	printWindow("����������1", z1);
	cout << "�������� kompleksirB:" << entropy(z1, z1.size()) << endl;
	imwrite("kompleksirB.jpg", z1);

	//���������������� ��+R-�����
	Mat z2;
	z2 = R / 2 + a / 2;
	printWindow("����������1", z2);
	cout << "�������� kompleksirR:" << entropy(z2, z2.size()) << endl;
	imwrite("kompleksirR.jpg", z2);

	//���������������� ��+G-�����
	Mat z3;
	z3 = G / 2 + a / 2;
	printWindow("����������1", z3);
	cout << "�������� kompleksirG:" << entropy(z3, z3.size()) << endl;
	imwrite("kompleksirG.jpg", z3);

	//���������������� � RGB
	int y = IR_f.rows;
	int x = IR_f.cols;
	Mat Psevdo(Size(y, x), CV_8UC3);//������� ������� 3-� ���������� �����������
	vector<Mat> ch;
	split(Psevdo, ch);

	//�������� 1
	ch[2] = R;//R
	ch[1] = G;//G
	ch[0] = z1;//B
	Mat pr10;
	merge(ch,pr10);
	printWindow("����������10", pr10);
	imwrite("ptimer10.jpg", pr10);
	Mat Pscd10gr;
	cvtColor(pr10, Pscd10gr, COLOR_BGR2GRAY);
	cout << "�������� Pscd10 :" << entropy(Pscd10gr, Pscd10gr.size()) << endl;
	printWindow("���������� 10 � ��������� ������", Pscd10gr);
	imwrite("primer10gr.jpg", Pscd10gr);

	//�������� 2
	ch[2] = z2;//R
	ch[1] = z3;//G
	ch[0] = z1;//B
	Mat pr100;
	merge(ch, pr100);
	printWindow("����������100", pr100);
	imwrite("ptimer100.jpg", pr100);
	Mat Pscd100gr;
	cvtColor(pr100, Pscd100gr, COLOR_BGR2GRAY);
	cout << "�������� Pscd100 :" << entropy(Pscd100gr, Pscd100gr.size()) << endl;
	printWindow("���������� 100 � ��������� ������", Pscd100gr);
	imwrite("primer100gr.jpg", Pscd100gr);

	waitKey(0);

	return 0;
}