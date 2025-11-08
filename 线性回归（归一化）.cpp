#include <bits/stdc++.h>
using namespace std;
inline double read(){ //快读 
    double x = 0.0;
    int f = 1;
    char c = getchar();
    while(!isdigit(c) && c != '.'){
        if (c == '-') f = -1;
        c = getchar();
    }
    while(isdigit(c)) {
        x = x * 10 + (c - '0');
        c = getchar();
    }
    if(c == '.'){
        c = getchar();
        double decimal = 0.1;
        while(isdigit(c)){
            x += (c-'0')*decimal;
            decimal *= 0.1;
            c = getchar();
        }
    }
    return x*f;
}
int n;
double tmp[100001],pre[100001];
vector<double> true_w; //实际权重 
vector<double> now_w; //现在权重 
double now_b; //现在偏差 
vector<vector<double> > simple_x; //示例x 
vector<double> simple_y; //示例y 
inline double lossfunction(vector<double> w,double b,bool kind,int x){ //导过的成本函数？ 
//	vector<double> y;
	if (kind == 1){ //w
		double sum = 0;
		for (int i=0;i<100;i++){
			pre[i] = b;
		} 
		for (int i=0;i<n;i++) {
			for (int j=0;j<100;j++){
				pre[j] += w[i]*simple_x[i][j];
			}
		}
		for (int i=0;i<100;i++){
			sum += (pre[i]-simple_y[i])*simple_x[x][i];
		}
		return sum/100;
	} 
	else{ //b
		double sum = 0;
		for (int i=0;i<100;i++){
			pre[i] = b;
		} 
		for (int i=0;i<n;i++) {
			for (int j=0;j<100;j++){
				pre[j] += w[i]*simple_x[i][j];
			}
		}
		for (int i=0;i<100;i++){
			sum += (pre[i]-simple_y[i]);
		}
		return sum/100;
	}
}
signed main(){
	srand(114514); //初始随机种子 
	n = read(); //权重个数 
	for (int i=0;i<n;i++){ //实际权重 
		double tmp = read();
		true_w.push_back(tmp);
	}
	double true_b = read(); //实际偏差 
	for (int i=0;i<n;i++){ //生成示例x 
		simple_x.push_back(vector<double>());
		double maxn = -200.0;
		double minn = 200.0;
		double num = 0;
		for (int j=0;j<100;j++){
			double tmp = (1+rand()%100) * 0.1;
			num+=tmp;
			maxn = max(maxn,tmp);
			minn = min(minn,tmp);
			simple_x[i].push_back(tmp);
		}
		num = num/100;//最大最小值归一 
		for (int j=0;j<100;j++){
			simple_x[i][j] = (simple_x[i][j]-minn)/(maxn-minn);
		}
	}
	for (int i=0;i<100;i++){
		tmp[i] = true_b + (1+rand()%100)*0.01;
	}
	for (int i=0;i<n;i++){ //生成示例y 
		for (int j=0;j<100;j++){
			tmp[j] += true_w[i]*simple_x[i][j];
		}
	}
	for (int j=0;j<100;j++){
		simple_y.push_back(tmp[j]);
	}
	/*
	for (int i=0;i<n;i++){
		for (int j=0;j<100;j++){
			printf("X_(%d,%d): %.2f,Y_(%d,%d): %.2f\n",i,j,simple_x[i][j],i,j,simple_y[j]);
		}
	}
	//示例输出
	*/
	int train_times = read(); //设置训练次数 
	for (int i=0;i<n;i++){
		now_w.push_back(0); 
	}
	now_b = 0;
	double lr = read(); //设置学习率 
	for (int i=1;i<=train_times;i++){
//		cout<<i<<endl; 
		vector<double> pre_w;
		for (int j=0;j<n;j++){
			double loss_w = lossfunction(now_w,now_b,1,j); //计算成本 (计算损失加反向传播)
			double pw = now_w[j]-lr*loss_w; 
			pre_w.push_back(pw);
		} 
		double loss_b = lossfunction(now_w,now_b,0,1); //计算成本 (计算损失加反向传播)
		double pre_b = now_b-lr*loss_b;
		for (int j=0;j<n;j++){
			now_w[j] = pre_w[j];
		}
		now_b = pre_b;
	} 
	for (int i=0;i<n;i++){
		printf("%.2lf ",now_w[i]); 
	}
	printf("%.2lf\n",now_b); 
	return 0;
}
/*
2           
3.0 5.0        
2.0            
1000          
0.01    
*/    
