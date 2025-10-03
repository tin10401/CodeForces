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

ll domino_solver(vll h, vll c) {
    // https://codeforces.com/contest/1131/problem/G?adcd1e=caf4fvg4ke61uy&csrf_token=55be3d76734d8ccd743640b909b92ea9
    h.insert(begin(h), 0);
    c.insert(begin(c), 0);
    const int N = h.size();
    vi L(N), R(N);
    {
        stack<int> s;
        for(ll i = N - 1; i >= 1; i--) {
            while(!s.empty() && s.top() - h[s.top()] >= i) {
                L[s.top()] = i + 1;
                s.pop();
            }
            s.push(i);
        }
        while(!s.empty()) {
            L[s.top()] = 1;
            s.pop();
        }
    }
    {
        stack<int> s;
        for(int i = 1; i < N; i++) {
            while(!s.empty() && s.top() + h[s.top()] <= i) {
                R[s.top()] = i - 1;
                s.pop();
            }
            s.push(i);
        }
        while(!s.empty()) {
            R[s.top()] = N - 1;
            s.pop();
        }
    }
    vll dp(N, INF);
    dp[0] = 0;
    stack<int> s;
    for(int i = 1; i < N; i++) {
        while(!s.empty() && R[s.top()] < i) s.pop();
        dp[i] = dp[L[i] - 1] + c[i];
        if(!s.empty()) dp[i] = min(dp[i], dp[s.top() - 1] + c[s.top()]);
        if(s.empty() || (dp[i - 1] + c[i] < dp[s.top() - 1] + c[s.top()])) s.push(i);
    }
    return dp.back();
}

vll h, c;
void solve() {
    cout << domino_solver(h, c) << '\n';
}

signed main() {
    IOS;
    startClock
    int t = 1;
    int n, m; cin >> n >> m;
    vvll a(n), b(n);
    for(int i = 0; i < n; i++) {
        int k; cin >> k;
        a[i].rsz(k);
        b[i].rsz(k);
        cin >> a[i] >> b[i];
    }
    int q; cin >> q;
    while(q--) {
        ll id, mul; cin >> id >> mul;
        id--;
        const int K = a[id].size();
        for(int i = 0; i < K; i++) {
            h.pb(a[id][i]);
            c.pb(b[id][i] * mul);
        }
    }
    //cin >> t;
    for(int i = 1; i <= t; i++) {   
        //cout << "Case #" << i << ": ";  
        solve();
    }
    endClock;
    printMemoryUsage();
    return 0;
}
