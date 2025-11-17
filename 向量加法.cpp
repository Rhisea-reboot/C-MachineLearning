//#pragma gcc -O3 -march=native add_avx2.c -o add_avx2
#include <immintrin.h> // AVX 头文件
#include <bits/stdc++.h>
using namespace std;
void add_arrays_avx2(float *a, float *b, float *c, int n) {
    int i = 0;
    // 处理对齐部分：每次 8 个 float（256 位向量）
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_load_ps(a + i);  // 加载 8 个 float 到向量寄存器
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);  // 向量加法
        _mm256_store_ps(c + i, vc);         // 存储结果
    }
    // 处理剩余数据（不足 8 个）
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
vector<float> a,b,c;
signed main(int argc,char *argv[]){
	int n = 1000;
	for (int i=0;i<n;i++){
		a.push_back((float)i);
		b.push_back((float)i*2);
		c.push_back(0.0);
	}
	add_arrays_avx2(a.data(),b.data(),c.data(),n);
	for (int i=0;i<n;i++){
		cout<<c[i]<<" ";
	}
	cout<<endl;
	return 0;
}
