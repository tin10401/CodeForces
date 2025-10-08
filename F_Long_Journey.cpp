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

ll gcd(ll a, ll b) { while (b != 0) { ll temp = b; b = a % b; a = temp; } return a; }
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }

void solve() {
    ll n, m; cin >> n >> m;
    vi a(n), b(n); cin >> a >> b;
    if(n == 1) {
        cout << (m < b[0] ? m : -1) << '\n';
        return;
    }
    ll l = 1;
    for(auto& x : a) l = lcm(x, l);
    auto move = [&](int r, int i) -> pii {
        if(r % a[i] == b[i]) return {(r + 1) % l, 1};
        if((r + 1) % a[i] == b[i]) return {r, 0};
        return {(r + 1) % l, 1};
    };
    const int K = 60;
    vvi par(l, vi(K));
    vvll add(l, vll(K));
    for(int r = 0; r < l; r++) {
        ll curr = r, turn = 0;
        for(int i = 0; i < n; i++) {
            auto [nxt, p] = move(curr, i);
            curr = nxt;
            turn += p;
        }
        if(r == 0 && turn == 0) {
            cout << -1 << '\n';
            return;
        }
        if(turn == 0) {
            turn = INF;
        }
        par[r][0] = curr;
        add[r][0] = turn;
    }
    for(int j = 1; j < K; j++) {
        for(int i = 0; i < l; i++) {
            int p = par[i][j - 1];
            par[i][j] = par[p][j - 1];
            add[i][j] = min(INF, add[i][j - 1] + add[p][j - 1]);
        }
    }
    ll res = 0, u = 0, turn = 0;
    for(int j = K - 1; j >= 0; j--) {
        if(add[u][j] && add[u][j] + res < m) {
            res += add[u][j];
            turn += (1LL << j) * n;
            u = par[u][j];
        }
    }
    for(int i = 0; res < m; i = (i + 1) % n) {
        auto [nxt, p] = move(u, i); 
        res += p;
        u = nxt;
        turn++;
        if(add[u][i] == INF) break;
    }
    cout << (res == m ? turn : -1) << '\n';
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
