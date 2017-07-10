#pragma once

#define POINTER_MATRIX

#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <omp.h>
#include <memory.h>
#include <memory>
#include <queue>
#include <random>

using namespace std;

extern random_device rd;
extern mt19937 eng;
extern uniform_int_distribution<> dst;

// ����� ��� ������ � ���������� ���������
////////////////////////////////////////////////////////////////////////////////
class Matrix2D {
private:
	int						RowCount;			// ����� �����
	int						ColCount;			// ����� ��������
#ifdef POINTER_MATRIX
	unique_ptr<double[]>	MatrixElements;		// ������ ��������� �������
#else
	vector<vector<double>>	MatrixElements;		// ������ ��������� �������
#endif

public:
	// �����������
#ifdef POINTER_MATRIX
	Matrix2D(void) : RowCount(0), ColCount(0), MatrixElements(nullptr) {};
	Matrix2D(const int &r, const int &c) : RowCount(r), ColCount(c) { 
		MatrixElements = unique_ptr<double[]>(new double[r*c]);
		memset(MatrixElements.get(), 0, r*c * sizeof(double)); 
	};
	Matrix2D(const vector<vector<double>> &v) {
		RowCount = v.size(); ColCount = (RowCount ? v[0].size() : 0);
		MatrixElements = unique_ptr<double[]>(new double[RowCount*ColCount]);
		int counter = 0;
		for (int i = 0; i < RowCount; ++i)
			for (int j = 0; j < ColCount; ++j) {
				MatrixElements.get()[counter++] = v[i][j];
			}
	};
	Matrix2D(const deque<vector<double>> &v) {
		RowCount = v.size(); ColCount = (RowCount ? v[0].size() : 0);
		MatrixElements = unique_ptr<double[]>(new double[RowCount*ColCount]);
		int counter = 0;
		for (int i = 0; i < RowCount; ++i)
			for (int j = 0; j < ColCount; ++j) {
				MatrixElements.get()[counter++] = v[i][j];
			}

	};
	Matrix2D(const Matrix2D &m) {
		ColCount = m.ColCount;
		RowCount = m.RowCount;
		MatrixElements = unique_ptr<double[]>(new double[RowCount*ColCount]);
		memcpy(MatrixElements.get(), m.MatrixElements.get(), ColCount*RowCount * sizeof(double));
	}
	Matrix2D& operator=(const Matrix2D &m) {
		ColCount = m.ColCount;
		RowCount = m.RowCount;
		MatrixElements = unique_ptr<double[]>(new double[RowCount*ColCount]);
		memcpy(MatrixElements.get(), m.MatrixElements.get(), ColCount*RowCount * sizeof(double));
		return *this;
	}
#else
	Matrix2D(void) : RowCount(0), ColCount(0) {};
	Matrix2D(const int &r, const int &c) : RowCount(r), ColCount(c), MatrixElements(vector<vector<double>>(r, vector<double>(c, 0.0))) {};
	Matrix2D(const vector<vector<double>> &v) : RowCount(v.size()), ColCount((v.size() ? v[0].size() : 0)), MatrixElements(v) {};
	Matrix2D(const deque<vector<double>> &v) {
		RowCount = v.size(); ColCount = (RowCount ? v[0].size() : 0);
		MatrixElements.resize(RowCount, vector<double>(ColCount, 0.0));
		for (int i = 0; i < RowCount; ++i)
			for (int j = 0; j < ColCount; ++j)
				MatrixElements[i][j] = v[i][j];
	};
#endif

	int						GetRowCount() const { return RowCount; }; // ����� �����
	int						GetColCount() const { return ColCount; }; // ����� ��������

#ifndef POINTER_MATRIX
	vector<vector<double>>	GetVector() { return MatrixElements; };
#endif

	double&					operator()(const int i, const int j); // �������� �������� �������
	double					operator()(const int i, const int j) const; // �������� �������� �������

	Matrix2D				operator+(const Matrix2D& m2); // �������� ��������
	Matrix2D				operator+(const double& v); // �������� �������� � ������
	Matrix2D				operator-(const Matrix2D& m2); // �������� ���������
	friend Matrix2D			operator-(const Matrix2D& m1); // �������� ����������
	Matrix2D				operator*(const Matrix2D& m2); // �������� ��������� ������
	friend Matrix2D			operator*(const double v, const Matrix2D& m2); // �������� ��������� ������� �� �����
	Matrix2D				operator*(const double& v); // �������� ��������� ������� �� �����
	Matrix2D				operator/(Matrix2D& m2); // �������� ������� ������
	Matrix2D				operator/(const double& v); // �������� ������� ������� �� �����

	double					Det(); // ���������� �������������
	Matrix2D				Inverse(); // ���������� �������� �������
	Matrix2D				Transpose(); // ���������� ����������������� �������
	double					Trace(); // ���������� ����� �������
	Matrix2D				LU() const; // ���������� LU-���������� �������
	Matrix2D				LUP(Matrix2D &P); // ���������� LUP-���������� �������
	void					SwapRows(int i, int j); // �������� i-� � j-� ����� ������� ������� 

													// ������� ��������� ������� � ���������� ������ � �������� ��������
													// ������ �� �������
#ifdef POINTER_MATRIX
	double* Matrix2DToPointer() { return MatrixElements.get(); };
#else
	vector<double> Matrix2DToVector() {
		vector<double> LineMatrixElements;
		for_each(MatrixElements.begin(), MatrixElements.end(), [&LineMatrixElements](vector<double> CurRow) {
			copy(CurRow.begin(), CurRow.end(), back_inserter(LineMatrixElements));
		});
		return LineMatrixElements;
	};
#endif
	void					ShowElements();
};
////////////////////////////////////////////////////////////////////////////////




// �������� ��� ������ ��� ���������
////////////////////////////////////////////////////////////////////////////////
Matrix2D CalculateOnes(int SizeNum);
#ifdef POINTER_MATRIX
Matrix2D PointerToMatrix2D(double *p, int NumRows, int NumCols);
#else
Matrix2D VectorToMatrix2D(vector<double> p, int NumRows, int NumCols);
#endif
////////////////////////////////////////////////////////////////////////////////


