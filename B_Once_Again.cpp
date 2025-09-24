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
const static ll INF = 4e18 + 10;
const static int inf = 1e9 + 100;
const static int MX = 1e5 + 5;

template<typename T>
struct Mat {
    int R, C;
    vt<vt<T>> a;
    T DEFAULT; 

    Mat(const vt<vt<T>>& m, T _DEFAULT = -inf) : R((int)m.size()), C(m.empty() ? 0 : (int)m[0].size()), a(m), DEFAULT(_DEFAULT) {}

    Mat(int _R, int _C, T _DEFAULT = -inf) : R(_R), C(_C), DEFAULT(_DEFAULT), a(R, vt<T>(C, _DEFAULT)) {}

    static Mat identity(int n, T _DEFAULT) {
        Mat I(n, n, _DEFAULT);
        for (int i = 0; i < n; i++)
            I.a[i][i] = T(0); // for min max do 0 instead of 1
        return I;
    }

    Mat operator*(const Mat& o) const {
        Mat r(R, o.C, DEFAULT);
        for(int i = 0; i < R; i++) {
            for(int k = 0; k < C; k++) {
                T v = a[i][k];
                for(int j = 0; j < o.C; j++) {
                    T w = o.a[k][j];
                    r.a[i][j] = max(r.a[i][j], v + o.a[k][j]);
                }
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
	
	bool operator==(const Mat& o) const {
        if(R != o.R || C != o.C) return false;
        for(int i = 0; i < R; i++)
            for(int j = 0; j < C; j++)
                if(a[i][j] != o.a[i][j]) return false;
        return true;
    }
};

void solve() {
    int n, t; cin >> n >> t;
    vi a(n); cin >> a;
    vvi A(n, vi(n, -inf));
    for(int s = 0; s < n; ++s) {
        for(int i = 0; i < n; ++i) {
            if(a[i] < a[s]) A[s][i] = -inf;
            else {
                A[s][i] = 1;
                for(int j = 0; j < i; ++j)
                    if(a[j] <= a[i])
                        A[s][i] = max(A[s][i], A[s][j] + 1);
            }
        }
    }
    auto it = Mat<int>(A).pow(t);
    int res = 0;
    for(auto& x : it.a) {
        for(auto& y : x) {
            res = max(res, y);
        }
    }
    cout << res << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
