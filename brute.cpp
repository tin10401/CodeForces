#line 2 "/Users/noya2/Desktop/Noya2_library/template/template.hpp"
using namespace std;

#include<bits/stdc++.h>
#line 1 "/Users/noya2/Desktop/Noya2_library/template/inout_old.hpp"
namespace noya2 {

template <typename T, typename U>
ostream &operator<<(ostream &os, const pair<T, U> &p){
    os << p.first << " " << p.second;
    return os;
}
template <typename T, typename U>
istream &operator>>(istream &is, pair<T, U> &p){
    is >> p.first >> p.second;
    return is;
}

template <typename T>
ostream &operator<<(ostream &os, const vector<T> &v){
    int s = (int)v.size();
    for (int i = 0; i < s; i++) os << (i ? " " : "") << v[i];
    return os;
}
template <typename T>
istream &operator>>(istream &is, vector<T> &v){
    for (auto &x : v) is >> x;
    return is;
}

void in() {}
template <typename T, class... U>
void in(T &t, U &...u){
    cin >> t;
    in(u...);
}

void out() { cout << "\n"; }
template <typename T, class... U, char sep = ' '>
void out(const T &t, const U &...u){
    cout << t;
    if (sizeof...(u)) cout << sep;
    out(u...);
}

template<typename T>
void out(const vector<vector<T>> &vv){
    int s = (int)vv.size();
    for (int i = 0; i < s; i++) out(vv[i]);
}

struct IoSetup {
    IoSetup(){
        cin.tie(nullptr);
        ios::sync_with_stdio(false);
        cout << fixed << setprecision(15);
        cerr << fixed << setprecision(7);
    }
} iosetup_noya2;

} // namespace noya2
#line 1 "/Users/noya2/Desktop/Noya2_library/template/const.hpp"
namespace noya2{

const int iinf = 1'000'000'007;
const long long linf = 2'000'000'000'000'000'000LL;
const long long mod998 =  998244353;
const long long mod107 = 1000000007;
const long double pi = 3.14159265358979323;
const vector<int> dx = {0,1,0,-1,1,1,-1,-1};
const vector<int> dy = {1,0,-1,0,1,-1,-1,1};
const string ALP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const string alp = "abcdefghijklmnopqrstuvwxyz";
const string NUM = "0123456789";

void yes(){ cout << "Yes\n"; }
void no(){ cout << "No\n"; }
void YES(){ cout << "YES\n"; }
void NO(){ cout << "NO\n"; }
void yn(bool t){ t ? yes() : no(); }
void YN(bool t){ t ? YES() : NO(); }

} // namespace noya2
#line 2 "/Users/noya2/Desktop/Noya2_library/template/utils.hpp"

#line 6 "/Users/noya2/Desktop/Noya2_library/template/utils.hpp"

namespace noya2{

unsigned long long inner_binary_gcd(unsigned long long a, unsigned long long b){
    if (a == 0 || b == 0) return a + b;
    int n = __builtin_ctzll(a); a >>= n;
    int m = __builtin_ctzll(b); b >>= m;
    while (a != b) {
        int mm = __builtin_ctzll(a - b);
        bool f = a > b;
        unsigned long long c = f ? a : b;
        b = f ? b : a;
        a = (c - b) >> mm;
    }
    return a << std::min(n, m);
}

template<typename T> T gcd_fast(T a, T b){ return static_cast<T>(inner_binary_gcd(std::abs(a),std::abs(b))); }

long long sqrt_fast(long long n) {
    if (n <= 0) return 0;
    long long x = sqrt(n);
    while ((x + 1) * (x + 1) <= n) x++;
    while (x * x > n) x--;
    return x;
}

template<typename T> T floor_div(const T n, const T d) {
    assert(d != 0);
    return n / d - static_cast<T>((n ^ d) < 0 && n % d != 0);
}

template<typename T> T ceil_div(const T n, const T d) {
    assert(d != 0);
    return n / d + static_cast<T>((n ^ d) >= 0 && n % d != 0);
}

template<typename T> void uniq(std::vector<T> &v){
    std::sort(v.begin(),v.end());
    v.erase(unique(v.begin(),v.end()),v.end());
}

template <typename T, typename U> inline bool chmin(T &x, U y) { return (y < x) ? (x = y, true) : false; }

template <typename T, typename U> inline bool chmax(T &x, U y) { return (x < y) ? (x = y, true) : false; }

template<typename T> inline bool range(T l, T x, T r){ return l <= x && x < r; }

} // namespace noya2
#line 8 "/Users/noya2/Desktop/Noya2_library/template/template.hpp"

#define rep(i,n) for (int i = 0; i < (int)(n); i++)
#define repp(i,m,n) for (int i = (m); i < (int)(n); i++)
#define reb(i,n) for (int i = (int)(n-1); i >= 0; i--)
#define all(v) (v).begin(),(v).end()

using ll = long long;
using ld = long double;
using uint = unsigned int;
using ull = unsigned long long;
using pii = pair<int,int>;
using pll = pair<ll,ll>;
using pil = pair<int,ll>;
using pli = pair<ll,int>;

namespace noya2{

/*　~ (. _________ . /)　*/

}

using namespace noya2;


#line 2 "c.cpp"


void solve(){
    ll ini; in(ini);
    const int mx = 301;
    // const int mx = 10;
    vector<pii> ab = [&]{
        int n; in(n);
        vector<int> mb(mx,-1);
        rep(i,n){
            int a, b; in(a,b);
            chmax(mb[a],b);
        }
        vector<pii> ret;
        rep(i,mx){
            if (mb[i] != -1){
                ret.emplace_back(i,mb[i]);
            }
        }
        return ret;
    }();
    const int lim = mx*mx;
    vector<int> f(lim);
    rep(y,lim){
        for (auto [a, b] : ab){
            if (y >= b){
                int x = y + a - b;
                if (x < lim){
                    chmax(f[x],f[y]+b);
                }
            }
        }
    }
    // out(f);
    ll ans = 0;
    rep(y,lim){
        for (auto [a, b] : ab){
            if (y < b) continue;
            if (y > ini) continue;
            ll k = (ini - y) / (a - b);
            chmax(ans,f[y]+k*b);
        }
    }
    ans += ini;
    out(ans);
}

int main(){
    int t = 1; //in(t);
    while (t--) { solve(); }
}

