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
const static int MX = 1e6 + 5;

#define M_PI 3.14159265358979323846
const static string pi = "3141592653589793238462643383279";
ll gcd(ll a, ll b) { while (b != 0) { ll temp = b; b = a % b; a = temp; } return a; }
ll lcm(ll a, ll b) { return (a / gcd(a, b)) * b; }
ll floor(ll a, ll b) { if(b < 0) a = -a, b = -b; if (a >= 0) return a / b; return a / b - (a % b ? 1 : 0); }
ll ceil(ll a, ll b) { if (b < 0) a = -a, b = -b; if (a >= 0) return (a + b - 1) / b; return a / b; }
int pct(ll x) { return __builtin_popcountll(x); }
int have_bit(ll x, int b) { return (x >> b) & 1; }
int min_bit(ll x) { return __builtin_ctzll(x); }
int max_bit(ll x) { return 63 - __builtin_clzll(x); } 
const vvi dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}}; // UP, DOWN, LEFT, RIGHT
const vvi knight_dirs = {{-2, -1}, {-2,  1}, {-1, -2}, {-1,  2}, {1, -2}, {1,  2}, {2, -1}, {2,  1}}; // knight dirs
const string dirChar = {'U', 'D', 'L', 'R'};
int modExpo(ll base, ll exp, ll mod) { ll res = 1; base %= mod; while(exp) { if(exp & 1) res = (res * base) % mod; base = (base * base) % mod; exp >>= 1; } return res; }
ll extended_gcd(ll a, ll b, ll &x, ll &y) { if (b == 0) { x = 1; y = 0; return a; } ll d = extended_gcd(b, a % b, y, x); y -= (a / b) * x; return d; }
int modExpo_on_string(ll a, string exp, int mod) { ll b = 0; for(auto& ch : exp) b = (b * 10 + (ch - '0')) % (mod - 1); return modExpo(a, b, mod); }
ll sum_even_series(ll n) { return (n / 2) * (n / 2 + 1);} 
ll sum_odd_series(ll n) { ll m = (n + 1) / 2; return m * m; }
ll sum_of_square(ll n) { return n * (n + 1) * (2 * n + 1) / 6; } // sum of 1 + 2 * 2 + 3 * 3 + 4 * 4 + ... + n * n
string make_lower(const string& t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return tolower(c); }); return s; }
string make_upper(const string&t) { string s = t; transform(all(s), s.begin(), [](unsigned char c) { return toupper(c); }); return s; }
template<typename T> T geometric_sum(ll n, ll k) { return (1 - T(n).pow(k + 1)) / (1 - n); } // return n^1 + n^2 + n^3 + n^4 + n^5 + ... + n^k
template<typename T> T geometric_power(ll p, ll k) { return (T(p).pow(k + 1) - 1) / T(p - 1); } // p^0 + p^1 + p^2 + p^3 + ... + p^k
template<typename T> T geometric_power_range(T base, ll startExp, ll endExp) { // return base^startExp + base^(startExp + 1) + ... + base^endExp
    if(startExp > endExp) return 0;
    T first = base.pow(startExp);
    ll len = endExp - startExp + 1;
    return first * (base.pow(len) - 1) / (base - 1);
}
bool is_perm(ll sm, ll square_sum, ll len) {return sm == len * (len + 1) / 2 && square_sum == len * (len + 1) * (2 * len + 1) / 6;} // determine if an array is a permutation base on sum and square_sum
//bool is_two_prime_sum(ll n) { return n >= 4 && (n % 2 == 0 || isPrime(n - 2)); }
//bool is_three_prime_sum(ll n) { return n >= 6 && (n % 2 || is_two_prime_sum(n - 2)); }
ll sqrt(ll n) { ll t = sqrtl(n); while(t * t < n) t++; while(t * t > n) t--; return t;}

vi DIV[MX];
int spf[MX];

void nt_processing() {  
    for(int i = 2; i < MX; i++) {
        if(spf[i] == 0) {
            for(int j = i; j < MX; j += i) {    
                if(spf[j] == 0) spf[j] = i; 
            }
        }
    }
} static const bool _nt_init = []() { nt_processing(); return true; }();

vi factorize(int n) {
    auto& divs = DIV[n];
    if(!divs.empty()) return divs;
    divs.pb(1);
    while(n > 1) {
        int x = spf[n];
        int cnt = 0;
        while(n % x == 0) {
            n /= x;
            cnt++;
        }
        int d = 1;
        int N = divs.size();
        while(cnt--) {
            d *= x;
            for(int i = 0; i < N; i++) divs.pb(divs[i] * d);
        }
    }
	srt(divs);
    return divs;
}

vvi rotate90(const vvi matrix) {
    int n = matrix.size(), m = matrix[0].size();
    vvi res(m, vi(n));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            res[j][n - 1 - i] = matrix[i][j];
    return res;
}

ll cnt[MX];
void solve() {
    int n, m; cin >> n >> m;
    vvi a(n, vi(m)); cin >> a;
    int rotate = false;
    vvll b(n, vll(m));
    int nn = n, mm = m;
    if(n > m) {
        a = rotate90(a);
        swap(n, m);
        rotate = true;
    }
    int q; cin >> q;
    vall(5) Q(q);
    for(auto& [x, y, d, c, s] : Q) {
        cin >> x >> y >> d >> c >> s;
        x--, y--;
        if(rotate) {
            tie(x, y) = make_tuple(y, nn - x - 1);
        }
    }
    vvpll in(m), out(m);
    for(ll i = 0; i < n; i++) {
        for(auto& [x, y, d, c, s] : Q) {
            ll X = (x - i) * (x - i);
            ll rem = d * d - X;
            if(rem >= 0) {
                // (j - y) ^ 2 <= rem
                // (j - y) <= abs(sqrt(rem))
                // -sq <= j - y <= sq
                ll sq = sqrt(rem);
                ll l = max(y - sq, 0LL);
                ll r = min((ll)m - 1, y + sq);
                if(l <= r) {
                    in[l].pb({c, s});
                    out[r].pb({c, s});
                }
            }
        }
        ll total = 0;
        for(ll j = 0; j < m; j++) {
            for(auto& [C, s] : in[j]) cnt[C] += s, total += s;
            int r = i, c = j;
            if(rotate) {
                r = nn - j - 1;
                c = i;
            }
            if(a[i][j] == 0) {
                b[r][c] = total; 
            } else {
                for(auto& d : factorize(a[i][j])) {
                    b[r][c] += cnt[d];
                }
            }
            for(auto& [C, s] : out[j]) cnt[C] -= s, total -= s;
            in[j].clear();
            out[j].clear();
        }
    }
    for(auto& x : b) {
        for(auto& y : x) cout << y << ' ';
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
