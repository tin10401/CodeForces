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
struct paint_grid {
    // https://codeforces.com/contest/1080/problem/C
    struct Paint { ll x1, y1, x2, y2; bool is_black; };

    int n, m;
    ll white, black;
    vector<Paint> paints;

    paint_grid(int _n, int _m) : n(_n), m(_m) {
        ll total = 1LL * n * m;
        white = (total + 1) / 2;
        black = total / 2;
    }

    static inline bool inter(ll x1, ll y1, ll x2, ll y2, ll a1, ll b1, ll a2, ll b2, ll &ix1, ll &iy1, ll &ix2, ll &iy2) {
        ix1 = max(x1, a1);
        iy1 = max(y1, b1);
        ix2 = min(x2, a2);
        iy2 = min(y2, b2);
        return ix1 <= ix2 && iy1 <= iy2;
    }

    static inline ll area(ll x1, ll y1, ll x2, ll y2) {
        return (x2 - x1 + 1) * 1LL * (y2 - y1 + 1);
    }

    static inline pll count_white_black_base(ll x1, ll y1, ll x2, ll y2) {
        ll h = y2 - y1 + 1;
        ll w = x2 - x1 + 1;
        ll a = h * w;
        ll whites = a / 2;
        ll blacks = a / 2;
        if(a & 1) {
            if((x1 + y1) & 1) blacks++;
            else whites++;
        }
        return {whites, blacks};
    }

    ll unique_area_from(int i, ll x1, ll y1, ll x2, ll y2, int last) {
        ll ix1, iy1, ix2, iy2;
        if(!inter(paints[i].x1, paints[i].y1, paints[i].x2, paints[i].y2, x1, y1, x2, y2, ix1, iy1, ix2, iy2)) return 0;
        ll res = area(ix1, iy1, ix2, iy2);
        for(int j = i + 1; j <= last; ++j)
            res -= unique_area_from(j, ix1, iy1, ix2, iy2, last);
        return res;
    }

    ll unique_base_white_from(int i, ll x1, ll y1, ll x2, ll y2, int last) {
        ll ix1, iy1, ix2, iy2;
        if(!inter(paints[i].x1, paints[i].y1, paints[i].x2, paints[i].y2, x1, y1, x2, y2, ix1, iy1, ix2, iy2)) return 0;
        auto [w0, b0] = count_white_black_base(ix1, iy1, ix2, iy2);
        ll res = w0;
        for(int j = i + 1; j <= last; ++j)
            res -= unique_base_white_from(j, ix1, iy1, ix2, iy2, last);
        return res;
    }

    void paint(ll x1, ll y1, ll x2, ll y2, bool is_black) {
        int last = (int)paints.size() - 1;

        auto [baseW, baseB] = count_white_black_base(x1, y1, x2, y2);
        ll white_in_rect = baseW;

        for(int i = 0; i <= last; ++i) {
            ll bw = unique_base_white_from(i, x1, y1, x2, y2, last);
            ll ua = unique_area_from(i, x1, y1, x2, y2, last);
            if(paints[i].is_black)
                white_in_rect -= bw;
            else
                white_in_rect += (ua - bw);
        }

        ll rect_area = area(x1, y1, x2, y2);
        if(is_black) {
            white -= white_in_rect;
            black += white_in_rect;
        } else {
            ll black_in_rect = rect_area - white_in_rect;
            white += black_in_rect;
            black -= black_in_rect;
        }

        paints.pb({x1, y1, x2, y2, is_black});
    }

    void paint_white(ll x1, ll y1, ll x2, ll y2) { paint(x1, y1, x2, y2, false); }
    void paint_black(ll x1, ll y1, ll x2, ll y2) { paint(x1, y1, x2, y2, true); }
};

void solve() {
    int n, m; cin >> n >> m;
    paint_grid p(n, m);
    {
        int r1, c1, r2, c2; cin >> r1 >> c1 >> r2 >> c2;
        p.paint_white(r1, c1, r2, c2);
    }
    {
        int r1, c1, r2, c2; cin >> r1 >> c1 >> r2 >> c2;
        p.paint_black(r1, c1, r2, c2);
    }
    cout << p.white << ' ' << p.black << '\n';
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
