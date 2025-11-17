#include <bits/stdc++.h>
#include <immintrin.h>
using namespace std; 
inline void mul_arrays_avx2(float *a,float *b,float *c,int n){
	int i=0;
	for (;i<n-n%8;i+=8){
		__m256 va = _mm256_loadu_ps(a+i);
		__m256 vb = _mm256_loadu_ps(b+i);
		__m256 vc = _mm256_mul_ps(va,vb);
		_mm256_storeu_ps(c+i,vc);
	}
	for (;i<n;i++){
		c[i] = a[i]*b[i];
	}
}
signed main(int argc,char *argv[]){
	int n = 100;
	vector<float> a,b,c;
	for (int i=0;i<n;i++){
		a.push_back((float)i);
		b.push_back((float)i*2);
		c.push_back(0.0);
	}
	mul_arrays_avx2(a.data(),b.data(),c.data(),n);
	for (int i=0;i<n;i++){
		cout<<c[i]<<" ";
	}
	cout<<endl;
	return 0;
}
