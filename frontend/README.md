# Smart RetailX - Promotion Management System (Frontend Demo)

A React-based demonstration frontend for the Personalized Promotion Engine, showcasing admin and customer dashboards for managing promotional campaigns.

## Features

### Admin Dashboard

- **Product Selection**: Choose from sample products with real pricing
- **Discount Configuration**: Set discount percentages (5-50%)
- **Promotion Types**:
  - **Personalized**: AI-powered customer targeting based on buying probability
  - **Bulk**: Send promotions to all customers
- **Customer Targeting**: View AI-predicted top customers with buying probability scores
- **Promotion Management**: Track active promotions with usage statistics
- **Dashboard Metrics**: Revenue, orders, alerts, and forecast accuracy

### Customer Dashboard

- **Available Promotions**: View personalized and bulk offers
- **Promotion Details**: Product info, discounts, savings calculations
- **Redeem Offers**: Generate redemption codes for checkout
- **Loyalty Tracking**: Points balance and savings summary
- **Limited Offers**: See remaining quantities for exclusive deals

## Quick Start

### Installation

```bash
cd frontend
npm install
```

### Run Development Server

```bash
npm start
```

The application will open at `http://localhost:3000`

## Demo Navigation

Use the role switcher in the top-right corner to toggle between:

- **Admin View**: Create and manage promotions
- **Customer View**: Browse and redeem available offers

## Sample Data

The demo uses realistic sample data from the main dataset:

- 12 Products (Bakery, Snacks, Beverages, Personal Care, Grains)
- 10 Customers with loyalty scores and demographics
- 3 Pre-configured active promotions

## Technology Stack

- **React 18.2**: UI framework
- **React Router 6**: Client-side routing
- **Lucide React**: Icon library
- **Custom CSS**: Tailwind-inspired utility classes

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── AdminDashboard.js    # Admin promotion creation UI
│   │   └── CustomerDashboard.js # Customer promotion view
│   ├── data/
│   │   └── sampleData.js        # Sample products, customers, promotions
│   ├── App.js                    # Main app with routing
│   ├── App.css                   # App styles
│   ├── index.js                  # Entry point
│   └── index.css                 # Global styles
└── package.json
```

## How It Works

### Promotion Creation Flow (Admin)

1. Select a product from the dropdown
2. Set discount percentage using slider
3. Choose promotion type (Personalized or Bulk)
4. For personalized: Set available discount count
5. System displays AI-predicted target customers
6. Click "Send Promotion" to activate

### Promotion Redemption Flow (Customer)

1. View available promotions (filtered by customer ID)
2. See savings, pricing, and limited availability
3. Click "Redeem Offer" to generate code
4. Use code at checkout (simulated)

## AI Simulation

The demo simulates the promotion engine's ML predictions:

- Customers scored based on loyalty, price sensitivity
- Buying probability calculated for product-customer pairs
- Top N customers selected for personalized promotions

## Key Components

### AdminDashboard

- Product selection and discount configuration
- Promotion type toggle (Personalized/Bulk)
- AI-powered customer targeting preview
- Active promotions table with usage tracking

### CustomerDashboard

- Personalized promotion filtering
- Offer cards with savings calculations
- Redemption code generation
- Loyalty points display

## Production Deployment

For production builds:

```bash
npm run build
```

This creates an optimized build in the `build/` folder.

## Notes

- **No Backend**: This is a frontend-only demo with mock data
- **Local State**: Promotions stored in component state (resets on refresh)
- **Sample Customer**: Customer view defaults to customer ID 2 (Dinesh Silva)
- **Simulated ML**: Buying probability uses simplified scoring algorithm

## Future Enhancements

- Connect to actual promotion engine backend API
- Real-time notifications for customers
- Analytics dashboard with charts
- Multi-customer login system
- Promotion expiry and scheduling
- Email/SMS notification integration

## License

This is a demonstration project for the Personalized Promotion Engine research.
