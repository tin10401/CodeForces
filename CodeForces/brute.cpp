#include <bits/stdc++.h>
using namespace std;

#define MAX 605505
#define pb push_back
#define MP make_pair

/*Suffix Array n logn^2 */
struct mytuple {
    int originalIdx;
    int firstHalf,secondHalf;
};

int sr[35][MAX]; //suffixRank , gives rank of jth suffix on ith iteration
int SA[MAX];
mytuple t[MAX];

bool compare(const mytuple &a , const mytuple &b) {
    return a.firstHalf == b.firstHalf ?a.secondHalf < b.secondHalf :a.firstHalf < b.firstHalf;
}

void getSA(string s, int length ) {
    int pos = 0;
    memset(SA , 0 ,sizeof(SA));

    for(int i=0; i<length; i++)
        sr[0][i] = s[i] - 'a'; 

    for(int cnt = 1,stp = 1; (cnt>>1) < length ; cnt<<=1,stp++) {
        for(int i = 0; i<length ; i++) {
            t[i].firstHalf = sr[stp-1][i];
            t[i].secondHalf = i+cnt < length ? sr[stp-1][i+cnt] : -1;
            t[i].originalIdx = i;
        }

        sort(t,t+length,compare);

        int rnk = 0;
        sr[stp][t[0].originalIdx] = 0;
        for(int i = 1 ; i<length ; i++) {
            if(t[i-1].firstHalf == t[i].firstHalf && t[i].secondHalf == t[i-1].secondHalf)
                rnk = sr[stp][t[i-1].originalIdx]; //same as the last one
            else
                rnk = i; //new rank
            sr[stp][t[i].originalIdx] = rnk;
        }
    }

    for(int i=0; i<length; i++)
        SA[i] = t[i].originalIdx;
    return ;
}
//return the LCP BETWEEN TWO IDX
int getLCP(int i,int j,int n) {
    int res = 0;
    if(i==j)
        return n - i;
    for(int stp = ceil(log(n)/log(2)) ; stp>=0 && i < n && j < n ; stp--){
        if(sr[stp][i] == sr[stp][j]) {
            res += 1<<stp;
            i += 1<<stp;
            j += 1<<stp;
        }
    }
    return res;
}

vector<string> in;
int posInSA[MAX]; //position of the ith string in SA
int invMapping[MAX];

int upSearch(int idx,int n,int lcp){
    int l = 0,r=idx;
    while(l<r){
        int md = (l+r)/2;
        if(getLCP(SA[md],SA[idx],n) < lcp)
            l = md+1;
        else
            r = md;
    }
    return l;
}


int downSearch(int idx,int n,int lcp){
    int l = idx,r=n-1;
    while(l<r){
        int md = (l+r+1)/2;
        if(getLCP(SA[md],SA[idx],n) < lcp)
            r = md-1;
        else
            l = md;
    }
    return l;
}

int main() {
    int n;
    cin>>n;
    for(int i=0;i<n;i++){
        string s;
        cin>>s;
        in.pb(s);
    }
    string all = in[0];
    for(int i=1;i<n;i++){
        all += '#';
        all += in[i];
    }
    
    //cout<<all<<endl;
   
    getSA(all,all.size());
    for(int i=0;i<all.size();i++)
        invMapping[SA[i]]=i;
    int cur = 0;
    for(int i=0;i<n;i++){
        posInSA[i]=invMapping[cur];
        cur += in[i].size()+1;
    }

/*    for(int i=0;i<all.size();i++)
        cout<<SA[i]<<" ";
    cout<<endl;

    for(int i=0;i<n;i++)
        cout<<posInSA[i]<<" ";
    cout<<endl;*/

    int q;
    cin>>q;
    while(q--){
        int a,b;
        cin>>a>>b;
        a--;
        b--;
        
        int ft = posInSA[a];
        int st = posInSA[b];
        if(ft>st)
            swap(ft,st);
       
        int lcp = getLCP(SA[ft],SA[st],all.size());
        lcp = min(lcp,(int)in[a].size());
        lcp = min(lcp,(int)in[b].size());
        //cout<<lcp<<endl;

        if(lcp==0){
            cout<<"0\n";
            continue;
        }
    
        int upIdx = upSearch(ft,all.size(),lcp);
        //cout<<upIdx<<endl;
        
        int downIdx = downSearch(st,all.size(),lcp);
        //cout<<downIdx<<endl;
        
        cout<<downIdx-upIdx+1<<"\n";
    }
    return 0;
}
