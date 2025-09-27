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

template<typename T, typename V = string>
vt<pair<T, int>> encode(const V& s) {
    vt<pair<T, int>> seg;
    for(auto& ch : s) {
        if(seg.empty() || ch != seg.back().ff) seg.pb({ch, 1});
        else seg.back().ss++;
    }
    return seg;
}
void solve() {
    int a, b, c, d; cin >> a >> b >> c >> d;
    string s; cin >> s;
    int n = s.size();
    int A = count(all(s), 'A');
    int B = count(all(s), 'B');
    A -= c + d;
    B -= c + d;
    if(A != a || B != b) {
        cout << "NO" << '\n';
        return;
    }
    vi ab, ba, both;
    for(int i = 0; i < n;) {
        int j = i + 1;
        while(j < n && s[j] != s[j - 1]) j++;
        int len = j - i;
        if(len > 1) {
            if(len & 1) both.pb(len / 2);
            else if(s[i] == 'A') ab.pb(len / 2);
            else ba.pb(len / 2);
        }
        i = j;
    }
    srtR(ab);
    srtR(ba);
    int cab = sum(ab), cba = sum(ba);
    for(auto& x : both) {
        int t = max(min(x, c - cab), 0);
        cab += t;
        cba += x - t;
    }
    // abababab
    // we want to lose as much -1 as possible
    for(auto& x : ab) {
        if(x > 1 && cab > c + 1 && cba < d) {
            x--;
            cab--;
            int t = min({d - cba, cab - c, x});
            cab -= t;
            cba += t;
        }
    }
    for(auto& x : ba) {
        if(x > 1 && cba > d + 1 && cab < c) {
            x--;
            cba--;
            int t = min({c - cab, cba - d, x});
            cba -= t;
            cab += t;
        }
    }
    cout << (cab >= c && cba >= d ? "YES" : "NO") << '\n';
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
