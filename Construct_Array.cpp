#include<bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
using namespace std;
template<class T> using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
#define vt vector
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
#define db double
#define ld long db
#define ll long long
#define ull unsigned long long
#define vi vt<int>
#define vvi vt<vi>
#define vvvi vt<vvi>
#define pii pair<int, int>
#define vpii vt<pii>
#define vvpii vt<vpii>
#define vll vt<ll>  
#define vvll vt<vll>
#define pll pair<ll, ll>    
#define vpll vt<pll>
#define vvpll vt<vpll>
#define ar(x) array<int, x>
#define var(x) vt<ar(x)>
#define vvar(x) vt<var(x)>
#define al(x) array<ll, x>
#define vall(x) vt<al(x)>
#define vvall(x) vt<vall(x)>
#define vs vt<string>
#define pb push_back
#define ff first
#define ss second
#define rsz resize
#define sum(x) (ll)accumulate(all(x), 0LL)
#define srt(x) sort(all(x))
#define srtR(x) sort(allr(x))
#define srtU(x) sort(all(x)), (x).erase(unique(all(x)), (x).end())
#define rev(x) reverse(all(x))
#define MAX(a) *max_element(all(a)) 
#define MIN(a) *min_element(all(a))
#define SORTED(x) is_sorted(all(x))
#define ROTATE(a, p) rotate(begin(a), begin(a) + p, end(a))
#define i128 __int128
#define IOS ios_base::sync_with_stdio(false); cin.tie(0); cout.tie(0)
#if defined(LOCAL) && __has_include("debug.h")
  #include "debug.h"
#else
  #define debug(...)
  #define startClock
  #define endClock
  inline void printMemoryUsage() {}
#endif
template<class T> using max_heap = priority_queue<T>; template<class T> using min_heap = priority_queue<T, vector<T>, greater<T>>;
template<typename T, size_t N> istream& operator>>(istream& is, array<T, N>& arr) { for (size_t i = 0; i < N; i++) { is >> arr[i]; } return is; }
template<typename T, size_t N> istream& operator>>(istream& is, vector<array<T, N>>& vec) { for (auto &arr : vec) { is >> arr; } return is; }
template<typename T1, typename T2>  istream &operator>>(istream& in, pair<T1, T2>& input) { return in >> input.ff >> input.ss; }
template<typename T> istream &operator>>(istream &in, vector<T> &v) { for (auto &el : v) in >> el; return in; }
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
const static ll INF = 1e18;
const static int inf = 1e9 + 100;
const static int MX = 1005;

template <int MOD>
struct mod_int {
    int value;
    
    mod_int(ll v = 0) { value = int(v % MOD); if (value < 0) value += MOD; }
    
    mod_int& operator+=(const mod_int &other) { value += other.value; if (value >= MOD) value -= MOD; return *this; }
    mod_int& operator-=(const mod_int &other) { value -= other.value; if (value < 0) value += MOD; return *this; }
    mod_int& operator*=(const mod_int &other) { value = int((ll)value * other.value % MOD); return *this; }
    mod_int pow(ll p) const { mod_int ans(1), a(*this); while (p) { if (p & 1) ans *= a; a *= a; p /= 2; } return ans; }
    
    mod_int inv() const { return pow(MOD - 2); }
    mod_int& operator/=(const mod_int &other) { return *this *= other.inv(); }
    
    friend mod_int operator+(mod_int a, const mod_int &b) { a += b; return a; }
    friend mod_int operator-(mod_int a, const mod_int &b) { a -= b; return a; }
    friend mod_int operator*(mod_int a, const mod_int &b) { a *= b; return a; }
    friend mod_int operator/(mod_int a, const mod_int &b) { a /= b; return a; }
    
    bool operator==(const mod_int &other) const { return value == other.value; }
    bool operator!=(const mod_int &other) const { return value != other.value; }
    bool operator<(const mod_int &other) const { return value < other.value; }
    bool operator>(const mod_int &other) const { return value > other.value; }
    bool operator<=(const mod_int &other) const { return value <= other.value; }
    bool operator>=(const mod_int &other) const { return value >= other.value; }
    
    mod_int operator&(const mod_int &other) const { return mod_int((ll)value & other.value); }
    mod_int& operator&=(const mod_int &other) { value &= other.value; return *this; }
    mod_int operator|(const mod_int &other) const { return mod_int((ll)value | other.value); }
    mod_int& operator|=(const mod_int &other) { value |= other.value; return *this; }
    mod_int operator^(const mod_int &other) const { return mod_int((ll)value ^ other.value); }
    mod_int& operator^=(const mod_int &other) { value ^= other.value; return *this; }
    mod_int operator<<(int shift) const { return mod_int(((ll)value << shift) % MOD); }
    mod_int& operator<<=(int shift) { value = int(((ll)value << shift) % MOD); return *this; }
    mod_int operator>>(int shift) const { return mod_int(value >> shift); }
    mod_int& operator>>=(int shift) { value >>= shift; return *this; }

    mod_int& operator++() { ++value; if (value >= MOD) value = 0; return *this; }
    mod_int operator++(int) { mod_int temp = *this; ++(*this); return temp; }
    mod_int& operator--() { if (value == 0) value = MOD - 1; else --value; return *this; }
    mod_int operator--(int) { mod_int temp = *this; --(*this); return temp; }

    explicit operator ll() const { return (ll)value; }
    explicit operator int() const { return value; }
    explicit operator db() const { return (db)value; }

    friend mod_int operator-(const mod_int &a) { return mod_int(0) - a; }
    friend ostream& operator<<(ostream &os, const mod_int &a) { os << a.value; return os; }
    friend istream& operator>>(istream &is, mod_int &a) { ll v; is >> v; a = mod_int(v); return is; }
};

const static int MOD = 1e9 + 7;
using mint = mod_int<MOD>;
using vmint = vt<mint>;
using vvmint = vt<vmint>;
using vvvmint = vt<vvmint>;
using pmm = pair<mint, mint>;
using vpmm = vt<pmm>;

int spf[MX];
vpii DIV[MX];

void nt_processing() {  
    for(int i = 2; i < MX; i++) {
        if(spf[i] == 0) {
            for(int j = i; j < MX; j += i) {    
                if(spf[j] == 0) spf[j] = i; 
            }
        }
    }
} static const bool _nt_init = []() { nt_processing(); return true; }();

vpii factorize(int n) {
    auto& divs = DIV[n];
    if(!divs.empty()) return divs;
    while(n > 1) {
        int x = spf[n];
        int cnt = 0;
        while(n % x == 0) {
            n /= x;
            cnt++;
        }
        divs.pb({x, cnt});
    }
	srt(divs);
    return divs;
}

template<typename T>
struct Mat {
    int R, C;
    vt<vt<T>> a;
    T DEFAULT; 

    Mat(const vt<vt<T>>& m, T _DEFAULT = 0) : R((int)m.size()), C(m.empty() ? 0 : (int)m[0].size()), a(m), DEFAULT(_DEFAULT) {}

    Mat(int _R, int _C, T _DEFAULT = 0) : R(_R), C(_C), DEFAULT(_DEFAULT), a(R, vt<T>(C, _DEFAULT)) {}

    static Mat identity(int n, T _DEFAULT) {
        Mat I(n, n, _DEFAULT);
        for (int i = 0; i < n; i++)
            I.a[i][i] = T(1);
        return I;
    }

    Mat operator*(const Mat& o) const {
        Mat r(R, o.C, DEFAULT);
        for(int i = 0; i < R; i++) {
            for(int k = 0; k < C; k++) {
                T v = a[i][k];
                if(v == DEFAULT) continue;
                for(int j = 0; j < o.C; j++)
                    r.a[i][j] = r.a[i][j] + v * o.a[k][j];
            }
        }
        return r;
    }

    Mat pow(ll e) const {
        Mat res = identity(R, DEFAULT), base = *this;
        while(e > 0) {
            if(e & 1) res = res * base;
            base = base * base;
            e >>= 1;
        }
        return res;
    }

    friend ostream& operator<<(ostream& os, const Mat& M) {
        for(int i = 0; i < M.R; i++) {
            for(int j = 0; j < M.C; j++) {
                os << M.a[i][j];
                if(j + 1 < M.C) os << ' ';
            }
            if(i + 1 < M.R) os << '\n';
        }
        return os;
    }
};

void solve() {
    ll n, m; cin >> n >> m;
    if(n <= 2) {
        cout << mint(m).pow(n) << '\n';
        return;
    }
    mint same = m, diff = (mint)m * (m - 1);
    vvmint B = {{same}, {diff}};
    vvmint A = {{0, 1}, {m - 1, m - 1}};
//    for(int i = 3; i <= n; i++) {
//        mint nsame = diff;
//        mint ndiff = same * (m - 1) + diff * (m - 1);
//        same = nsame;
//        diff = ndiff;
//    }
    auto it = (Mat<mint>(A).pow(n - 2) * B);
    mint res = it.a[0][0] + it.a[1][0];
    cout << res << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
