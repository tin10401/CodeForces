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

struct non_monotone_triples { // return the longest chain without monotone triples in [l, r]
    // https://codeforces.com/contest/1332/problem/G
    vi a;
    int n;
    var(3) a3;
    var(4) a4;
    non_monotone_triples(const vi& _a) : n(_a.size()), a(_a) {
        a3.assign(n, {-1, -1, -1});
        a4.assign(n, {-1, -1, -1, -1});
        build();
    }

    void build() {
        set<int> non_highs, non_lows, neither;
        vi highs, lows, in_highs(n), in_lows(n);
        for(int i = 0; i < n; i++) {
            if(i) {
                a4[i] = a4[i - 1];
                a3[i] = a3[i - 1];
            }
            while(!highs.empty() && a[highs.back()] < a[i]) {
                int j = highs.back(); highs.pop_back();
                non_highs.insert(j);
                in_highs[j] = false;
                if(!in_lows[j]) neither.insert(j);
            } 
            while(!lows.empty() && a[lows.back()] > a[i]) {
                int j = lows.back(); lows.pop_back();
                non_lows.insert(j);
                in_lows[j] = false;
                if(!in_highs[j]) neither.insert(j);
            }
            in_lows[i] = in_highs[i] = true;
            highs.pb(i);
            lows.pb(i);
            auto highs_it = lb(all(highs), i, [&](const int& x, const int& y) {return a[x] > a[y];});
            auto lows_it = lb(all(lows), i, [&](const int& x, const int& y) {return a[x] < a[y];});
            int last_high = highs_it == begin(highs) ? -1 : *prev(highs_it);
            int last_low = lows_it == begin(lows) ? -1 : *prev(lows_it);
            {
                auto it = neither.lb(min(last_high, last_low));
                if(it != begin(neither)) {
                    it--;
                    debug(lows, highs, neither, last_high, last_low, *it);
                    int mx = *lb(all(highs), *it);
                    int mn = *lb(all(lows), *it);
                    a4[i] = max(a4[i], {*it, min(mn, mx), max(mn, mx), i});
                }
            }
            {
                auto it = non_highs.lb(last_high);
                if(it != begin(non_highs)) {
                    it--;
                    int mx = *lb(all(highs), *it);
                    a3[i] = max(a3[i], {*it, mx, i});
                }
            }
            {
                auto it = non_lows.lb(last_low);
                if(it != begin(non_lows)) {
                    it--;
                    int mn = *lb(all(lows), *it);
                    a3[i] = max(a3[i], {*it, mn, i});
                }
            }
        }
    }

    vi query(int l, int r) {
        if(a4[r][0] >= l) {
            return {a4[r][0], a4[r][1], a4[r][2], a4[r][3]};
        }
        if(a3[r][0] >= l) {
            return {a3[r][0], a3[r][1], a3[r][2]};
        }
        return {};
    }
};

void solve() {
    int n, q; cin >> n >> q;
    vi a(n); cin >> a;
    non_monotone_triples nmt(a);
    while(q--) {
        int l, r; cin >> l >> r;
        l--, r--;
        auto ans = nmt.query(l, r);
        cout << ans.size() << '\n';
        for(auto& x : ans) {
            cout << x + 1 << ' ';
        }
        cout << '\n';
    }
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
