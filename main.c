#include <stdio.h>
#include <time.h>
#include <wmmintrin.h>

// sizeof(unsigned long long)==8
// GF(2^128):
// primitive(irreducable) polynomial is
// g(x)=x^128+x^7+x^2+x+1

//typedef unsigned long long int ull;


void print128_num(__m128i var)
 {
 int64_t *v64val = (int64_t*) &var;
 printf("%.16lli %.16lli\n", v64val[1], v64val[0]);
 }

unsigned int deg(const __m128i a)
 {
 unsigned int j=0;
 __m128i b=a;
 unsigned long long int* b1=&b;
 while(b1[1])
  {
  b1[1]>>=1;
  ++j;
  }
 if(j>0)
  return 64+j;
 while(b1[0])
  {
  b1[0]>>=1;
  ++j;
  }
 return j;
 }

__m128i first_mul_mod(__m128i a, __m128i b)
 {
 unsigned long long *a1=&a,*b1=&b,olda11;
 long long c1[2]={0,0};
 unsigned k=0;
 while(!(b1[0]==0&&b1[1]==0))
  {
  if(b1[0]&1)
   {
   c1[0]^=a[0];
   c1[1]^=a[1];
   }
//a=((a<<1)+_mm_set_epi64x(a1[0]&0x8000000000000000,0))^(a1[1]&0x8000000000000000?{0x87,0}:{0,0});
  olda11=a1[1];
  a1[1]=((a1[1]<<1)+(a1[0]&0x8000000000000000));
  a1[0]=(a1[0]<<1)^(olda11 & 0x8000000000000000 ? 0x87 : 0);

//b=(b>>1)+_mm_set_epi64x(0,(b1[1]&1)<<63);
  b[0]=(b[0]>>1)+b1[1]&1;
  b[1]>>=1;

  ++k;
  }
 __m128i c={c1[0],c1[1]};
 return c;
 }//9223372036854775808

__m128i second_mul_mod(const __m128i a, const __m128i b)
 {
/*1)сначала нужно получить коэффициенты c'(x)=a(x)b(x)
  2)разделить c'(x) на старшую часть с(x) и младшую h(x)
  3)найти по формуле p(x)=c(x)x^t (mod g(x))
  4)сложить p(x) и h(x) (заксорить коэфициенты)*/

// if imm8==17, то a1*b1
// if imm8==1,  то a1*b0
// if imm8==16, то a0*b1
// if imm8==0,  то a0*b0

 __m128i c=_mm_clmulepi64_si128(a,b,17),
         mid=_mm_clmulepi64_si128(a,b,1)+_mm_clmulepi64_si128(a,b,16),
         h=_mm_clmulepi64_si128(a,b,0);
 c+=_mm_srli_si128(mid,8);
 h+=_mm_slli_si128(mid,8);

//qz (q со звёздочкой) была заранее посчитана, и как не странно
//qz оказалось равно g(x), на всякий случай pz=x^14+x^4+x^2+1,
//deg(qz)=128, что не совсем подходит под тип __m128;
//поэтому qz делим на коэффициент qz_up=1 при x^128 и остальную часть qz_low
 __m128i qz_up={1,0}, qz_low={0x87,0};

 //long long int buf1[2]=_mm_clmulepi64_si128(c,qz_up,1);
 __m128i tmp_var1= _mm_slli_si128(_mm_clmulepi64_si128(c,qz_up,1),8);
                   //{0,buf1[0]};

 //long long int buf2[2]=_mm_clmulepi64_si128(c,qz_low,1);
 __m128i tmp_var2=_mm_srli_si128(_mm_clmulepi64_si128(c,qz_low,1),8);
                   //{buf2[1],0};

 __m128i c_mul_qz_1=tmp_var1+_mm_clmulepi64_si128(c,qz_up,0)+tmp_var2;

 __m128i tmp_var3=_mm_slli_si128(_mm_clmulepi64_si128(c,qz_low,1),8);
                   //{0,buf2[0]};
 long long *tv3_buf=&tmp_var3;
 __m128i c_mul_qz_0=_mm_set_epi64x(tv3_buf[0],0)+_mm_clmulepi64_si128(c,qz_low,0);

//ПРИВЕТ, КОСТЫЛЬ
 unsigned long long int* buf1=&c_mul_qz_1;
 unsigned long long int* buf0=&c_mul_qz_0;
 long long int buf[2];

 unsigned int d=deg(c_mul_qz_1);
 if(d>63)
  {
  d-=63;
  buf[0]=buf0[1]>>d+buf1[0]<<(64-d);
  buf[1]=buf1[0]>>d+buf1[1]<<(64-d);
  }
 else
  if(d==63)
   {
   buf[0]=buf0[1];
   buf[1]=buf1[0];
   }
  else
   if(d>0)
    {
    buf[0]=buf0[0]>>d+buf0[1]<<(64-d);
    buf[1]=buf0[1]>>d+buf1[0]<<(64-d);
    }
   else
    {
    buf[0]=buf0[0];
    buf[1]=buf0[1];
    }
//ПОКА

 __m128i Mt={buf[0],buf[1]};

//т.к. gz==qz_low, дальше не будем вводить новую переменную, а
//будем использовать qz_low

// long long int buf5[2]=_mm_clmulepi64_si128(Mt,qz_low,1);
 __m128i tmp_var4=_mm_slli_si128(_mm_clmulepi64_si128(Mt,qz_low,1),8);
                   //{0,buf5[0]};

 c=tmp_var4+_mm_clmulepi64_si128(Mt,qz_low,0);

 return c+h;
 }


int main()
 {
 for(;;)
  {
  clock_t t1,t2;
  double dur;
  long long int buf1[2],buf2[2];

  printf("Input your polynoms: ");
  scanf("%lli %lli",&(buf1[1]),buf1);
  printf("a=(%lli,%lli)\nanother one: ",buf1[1],buf1[0]);
  scanf("%lli %lli",&(buf2[1]),buf2);
  printf("b=(%i,%lli)\n\n\tFIRST\n",buf2[1],buf2[0]);

          //big-endian!
  __m128i a={buf1[0],buf1[1]},b={buf2[0],buf2[1]},c;
  t1=clock();
  c=first_mul_mod(a,b);
  t2=clock();
  dur=1000.0*(t2-t1)/CLOCKS_PER_SEC;
  print128_num(c);
  printf("CPU time used : %.2f ms\n\n\tSECOND\n", dur);

  t1=clock();
  c=second_mul_mod(a,b);
  t2=clock();
  dur=1000.0*(t2-t1)/CLOCKS_PER_SEC;
  print128_num(c);
  printf("CPU time used : %.2f ms\n\n", dur);
  }
 return 0;
 }
