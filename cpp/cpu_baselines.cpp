void matmul_cpu(const float*a,const float*b,float*c,int m,int n,int k){for(int r=0;r<m;++r)for(int col=0;col<n;++col){float s=0;for(int x=0;x<k;++x)s+=a[r*k+x]*b[x*n+col];c[r*n+col]=s;}}
