# 🏀 QuantBet Quick Start Guide

Daily NBL betting dashboard - run on your desktop with Docker.

## Prerequisites
- Docker Desktop installed and running
- Your Odds API key (get free at https://the-odds-api.com)

## 1. Setup (One Time)

Copy your API key to `.env`:
```bash
# In the QuantBet folder, edit .env file:
ODDS_API_KEY=your_api_key_here
ADMIN_USER=admin
ADMIN_PASS=your_password
```

## 2. Start the Dashboard

Open PowerShell in the project folder and run:
```bash
docker-compose up -d
```

Wait ~2 minutes for first build, then access:

| Service | URL |
|---------|-----|
| **Dashboard** | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Database Admin | http://localhost:5050 (profile: admin) |

## 3. Daily Workflow

1. Open http://localhost:3000
2. View "Today's Picks" section at top
3. See games with odds, edge, and Kelly stake
4. Place bets on bookmaker sites

## 4. Stop/Restart

```bash
# Stop
docker-compose down

# Restart
docker-compose up -d

# View logs
docker-compose logs -f api
```

## 5. Troubleshooting

**No games showing?**
- NBL season runs Oct-Feb
- WNBL runs Nov-Mar
- Outside season, empty is normal

**API error?**
- Check your ODDS_API_KEY in .env
- Free tier: 500 requests/month
- Check quota: http://localhost:8000/docs → /odds/quota

**Docker not starting?**
```bash
docker-compose down -v  # Reset volumes
docker-compose build --no-cache  # Rebuild
docker-compose up -d
```

---

That's it! Open http://localhost:3000 on game days to see your picks.
