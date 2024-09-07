#pragma GCC target("popcnt")
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace __gnu_pbds;
using namespace std;
typedef tree<pair<int, int>, null_type, less<pair<int, int>>, rb_tree_tag, tree_order_statistics_node_update> ordered_set;
#define all(x) begin(x), end(x)
#define allr(x) rbegin(x), rend(x)
#define ub upper_bound
#define lb lower_bound
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
    string s; cin >> s;
    map<ar(26), int> mp;
    ar(26) vis = {}, cnt = {};
    for(auto& ch : s) vis[ch - 'a']++;
    auto containEach = [&]()
    {
        for(int i = 0; i < 26; i++)
        {
            if(vis[i] && cnt[i] <= 0) return false;
        }
        return true;
    };

    int res = 0;
    mp[cnt]++;
    for(auto& ch : s)
    {
        cnt[ch - 'a']++;
        if(containEach())
        {
            for(int i = 0; i < 26; i++) if(cnt[i]) cnt[i]--;
        }
        res += mp[cnt]++;
    }
    cout << res << endl;
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr); 
    solve();
    return 0;
}

