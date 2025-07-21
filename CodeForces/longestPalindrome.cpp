#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ll unsigned long long
#define int long long
#define pb push_back
#define vi vector<int>
#define pii pair<int, int>
#define vpii vector<pair<int, int>>
#define f first
#define s second
#define ar(x) array<int, x>
const static int INF = 1LL << 61;
const static int MOD = 1e9 + 7;
const static string no = "NO\n";
const static string yes = "YES\n";
const vector<vector<int>> dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};

int modExpo(int base, int exp, int mod)
{
    int res = 1;
    while(exp)
    {
        if(exp & 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return res;
}

void solve()
{
    string s;
    cin >> s;
    string temp;
    temp += '$';
    for(auto& ch : s) temp += ch, temp += '$';
    s = temp;
    int start = 0, len = 1, n = s.size();
    vector<int> prefix(n);
    for(int i = 0, right = -1, left = 0; i < n; i++)
    {
        int k = i > right ? 1 : min(prefix[left + right - i], right - i + 1);
        while(i - k >= 0 && i + k < n && s[i - k] == s[i + k]) k++;
        prefix[i] = k--;
//        cout << prefix[i] << " ";
        if(2 * k > len)
        {
            len = k * 2;
            start = i - k;
        }
        if(i + k > right) right = i + k, left = i - k;
    }
    temp = s.substr(start, len);
    string ans;
    for(auto& ch : temp) if(ch != '$') ans += ch;
    cout << ans << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

