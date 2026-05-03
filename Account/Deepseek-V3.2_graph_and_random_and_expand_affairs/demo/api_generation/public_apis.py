import os
import yaml
from transformers import AutoTokenizer
from openai import OpenAI


def main():
    # === 输入参数 ===
    system_prompt = "You are a Python API generator. You need to create API functions based on the API function names and brief descriptions provided within the <apis></apis> tags (you can implement very simple logic internally), and then complete the LangChain format encapsulation."

    client = OpenAI(
        api_key="",  # 替换为你的API Key
        base_url="https://"  # API中转地址
    )
    # === 读取YAML并填充 ===
    user_prompt = '''
        You are a simple Python API synthesis expert.  
        Below, you will be provided with the current API function names and their brief descriptions:
        <apis>
        API_name: AdoptAPet | Description: Resource to help get pets adopted
        API_name: Axolotl | Description: Collection of axolotl pictures and facts
        API_name: Cat Facts | Description: Daily cat facts
        API_name: Cataas | Description: Cat as a service (cats pictures and gifs)
        API_name: AniAPI | Description: Anime discovery, streaming & syncing with trackers
        API_name: AniDB | Description: Anime Database
        API_name: AniList | Description: Anime discovery & tracking
        API_name: AnimeChan | Description: Anime quotes (over 10k+)
        API_name: AbuseIPDB | Description: IP/domain/URL reputation
        API_name: AlienVault Open Threat Exchange (OTX) | Description: IP/domain/URL reputation
        API_name: CAPEsandbox | Description: Malware execution and analysis
        API_name: Google Safe Browsing | Description: Google Link/Domain Flagging
        API_name: Améthyste | Description: Generate images for Discord users
        API_name: Art Institute of Chicago | Description: Art
        API_name: Colormind | Description: Color scheme generator
        API_name: ColourLovers | Description: Get various patterns, palettes and images
        API_name: Auth0 | Description: Easy to implement, adaptable authentication and authorization platform
        API_name: GetOTP | Description: Implement OTP flow quickly
        API_name: Micro User Service | Description: User management and authentication
        API_name: MojoAuth | Description: Secure and modern passwordless authentication platform
        API_name: Bitquery | Description: Onchain GraphQL APIs & DEX APIs
        API_name: Chainlink | Description: Build hybrid smart contracts with Chainlink
        API_name: Chainpoint | Description: Chainpoint is a global network for anchoring data to the Bitcoin blockchain
        API_name: Covalent | Description: Multi-blockchain data aggregator platform
        API_name: Bhagavad Gita | Description: Open Source Shrimad Bhagavad Gita API including 21+ authors translation in Sanskrit/English/Hindi        
        API_name: Bhagavad Gita telugu | Description: Bhagavad Gita API in telugu and odia languages
        API_name: Bible-api | Description: Free Bible API with multiple languages
        API_name: Ganjoor | Description: Classic Persian poetry works including access to related manuscripts, recitations and music tracks
        API_name: Apache Superset | Description: API to manage your BI dashboards and data sources on Superset
        API_name: Charity Search | Description: Non-profit charity data
        API_name: Clearbit Logo | Description: Search for company logos and embed them in your projects
        API_name: Domainsdb.info | Description: Registered Domain Names Search
        API_name: Public Holidays | Description: Data on national, regional, and religious holidays via API
        API_name: Calendarific | Description: Worldwide Holidays
        API_name: Checkiday - National Holiday API | Description: Industry-leading Holiday API. Over 5,000 holidays and thousands of descriptions. Trusted by the World’s leading companies
        API_name: Church Calendar | Description: Upload and share your files anonymously
        API_name: AnonFiles | Description: Upload and share your files
        API_name: Box | Description: File Sharing and Storage
        API_name: File.io | Description: Super simple file sharing, convenient, anonymous and secure
        API_name: Azure DevOps Health | Description: Resource health helps you diagnose and get support when an Azure issue impacts your resources
        API_name: Bitrise | Description: Build tool and processes integrations to create efficient development pipelines
        API_name: Buddy | Description: The fastest continuous integration and continuous delivery platform
        API_name: CircleCI | Description: Automate the software development process using continuous integration and continuous delivery
        API_name: apilayer coinlayer | Description: Real-time Crypto Currency Exchange Rates
        API_name: Binance | Description: Exchange for Trading Cryptocurrencies based in China
        API_name: Bitcambio | Description: Get the list of all traded assets in the exchange
        API_name: BitcoinAverage | Description: Digital Asset Price Data for the blockchain industry
        API_name: Amdoren | Description: Free currency API with over 150 currencies
        API_name: apilayer fixer.io | Description: Exchange rates and currency conversion
        API_name: Currency-api | Description: Free Currency Exchange Rates API with 150+ Currencies & No Rate Limits
        API_name: CurrencyFreaks | Description: Provides current and historical currency exchange rates with free plan 1K requests/month
        API_name: Postman Echo | Description: Test api server to receive and return value from HTTP method
        API_name: PurgoMalum | Description: Content validator against profanity & obscenity
        API_name: US Autocomplete | Description: Enter address data quickly with real-time address suggestions
        API_name: US Extract | Description: Extract postal addresses from any text including emails
        API_name: 24 Pull Requests | Description: Project to promote open source collaboration during December
        API_name: Screenshot | Description: Take programmatic screenshots of web pages from any website
        API_name: Agify.io | Description: Estimates the age from a first name
        API_name: API Gratis | Description: Multiples services and public APIs
        API_name: Chinese Character Web | Description: Chinese character definitions and pronunciations
        API_name: Chinese Text Project | Description: Online open-access digital library for pre-modern Chinese texts
        API_name: Collins | Description: Bilingual Dictionary and Thesaurus Data
        API_name: Free Dictionary | Description: Definitions, phonetics, pronounciations, parts of speech, examples, synonyms
        API_name: Asana | Description: Programmatic access to all data in your asana system
        API_name: ClickUp | Description: ClickUp is a robust, cloud-based project management tool for boosting productivity
        API_name: Clockify | Description: Clockify's REST-based API can be used to push/pull data to/from it & integrate it with other systems
        API_name: CloudConvert | Description: Online file converter for audio, video, document, ebook, archive, image, spreadsheet, presentation
        API_name: apilayer mailboxlayer | Description: Email address validation
        API_name: Email Validation | Description: Validate email addresses for deliverability and spam
        API_name: Cloudmersive Validate | Description: Validate email addresses, phone numbers, VAT numbers and domain names
        API_name: Disify | Description: Validate and detect disposable and temporary email addresses
        API_name: chucknorris.io | Description: JSON API for hand curated Chuck Norris jokes
        API_name: Corporate Buzz Words | Description: REST API for Corporate Buzz Words
        API_name: Excuser | Description: Get random excuses for various situations
        API_name: Fun Fact | Description: A simple HTTPS api that can randomly select and return a fact from the FFA database
        API_name: BreezoMeter Pollen | Description: Daily Forecast pollen conditions data for a specific location
        API_name: Carbon Interface | Description: API to calculate carbon (C02) emissions estimates for common C02 emitting activities
        API_name: Climatiq | Description: Calculate the environmental footprint created by a broad range of emission-generating activities
        API_name: Cloverly | Description: API calculates the impact of common carbon-intensive activities in real time
        API_name: Eventbrite | Description: Find events
        API_name: SeatGeek | Description: Search events, venues and performers
        API_name: Ticketmaster | Description: Search events, attractions, or venues
        API_name: VAT Validation | Description: Validate VAT numbers and calculate VAT rates
        API_name: Aletheia | Description: Insider trading data, earnings call analysis, financial statements, and more
        API_name: Alpaca | Description: Realtime and historical market data on all US equities and ETFs
        API_name: Alpha Vantage | Description: Realtime and historical stock data
        API_name: apilayer marketstack | Description: Real-Time, Intraday & Historical Market Data API
        API_name: Banco do Brasil | Description: All Banco do Brasil financial transaction APIs
        API_name: Bank Data API | Description: Instant IBAN and SWIFT number validation across the globe
        API_name: Billplz | Description: Payment platform
        API_name: Binlist | Description: Public access to a database of IIN/BIN information
        API_name: Boleto.Cloud | Description: A api to generate boletos in Brazil
        API_name: Citi | Description: All Citigroup account and statement data APIs
        API_name: Econdb | Description: Global macroeconomic data
        API_name: Fed Treasury | Description: U.S. Department of the Treasury Data
        API_name: Finage | Description: Finage is a stock, currency, cryptocurrency, indices, and ETFs real-time & historical data provider
        API_name: Financial Modeling Prep | Description: Realtime and historical stock data
        API_name: Finnhub | Description: Real-Time RESTful APIs and Websocket for Stocks, Currencies, and Crypto
        API_name: FRED | Description: Economic data from the Federal Reserve Bank of St. Louis
        API_name: Front Accounting APIs | Description: Front accounting is multilingual and multicurrency software for small businesses
        API_name: Hotstoks | Description: Stock market data powered by SQL
        API_name: IEX Cloud | Description: Realtime & Historical Stock and Market Data
        API_name: IG | Description: Spreadbetting and CFD Market Data
        API_name: Indian Mutual Fund | Description: Get complete history of India Mutual Funds Data
        API_name: Intrinio | Description: A wide selection of financial data feeds
        API_name: Klarna | Description: Klarna payment and shopping service
        API_name: MercadoPago | Description: Mercado Pago API reference - all the information you need to develop your integrations
        API_name: Mono | Description: Connect with users’ bank accounts and access transaction data in Africa
        API_name: Moov | Description: The Moov API makes it simple for platforms to send, receive, and store money
        API_name: Nordigen | Description: Connect to bank accounts using official bank APIs and get raw transaction data
        API_name: OpenFIGI | Description: Equity, index, futures, options symbology from Bloomberg LP
        API_name: Plaid | Description: Connect with user's bank accounts and access transaction data
        API_name: Polygon | Description: Historical stock market data
        API_name: Portfolio Optimizer | Description: Portfolio analysis and optimization
        API_name: Razorpay IFSC | Description: Indian Financial Systems Code (Bank Branch Codes)
        API_name: Real Time Finance | Description: Websocket API to access realtime stock data
        API_name: SEC EDGAR Data | Description: API to access annual reports of public US companies
        API_name: SmartAPI | Description: Gain access to set of and create end-to-end broking services
        API_name: StockData | Description: Real-Time, Intraday & Historical Market Data, News and Sentiment API
        API_name: Styvio | Description: Realtime and historical stock data and current stock sentiment
        API_name: Tax Data API | Description: Instant VAT number and tax validation across the globe
        API_name: Tradier | Description: US equity/option market data (delayed, intraday, historical)
        API_name: Twelve Data | Description: Stock market data (real-time & historical)
        API_name: WallstreetBets | Description: WallstreetBets Stock Comments Sentiment Analysis
        API_name: Yahoo Finance | Description: Real time low latency Yahoo Finance API for stock market, crypto currencies, and currency exchange
        API_name: YNAB | Description: Budgeting & Planning
        API_name: Zoho Books | Description: Online accounting software, built for your business
        API_name: BaconMockup | Description: Resizable bacon placeholder images
        API_name: Chomp | Description: 	Data about various grocery products and foods
        API_name: Open Brewery DB | Description: Breweries, Cideries and Craft Beer Bottle Shops
        API_name: Open Food Facts | Description: Food Products Database
        API_name: Age of Empires II | Description: Get information about Age of Empires II resources
        API_name: AmiiboAPI | Description: Nintendo Amiibo Information
        API_name: Autochess VNG | Description: Rest Api for Autochess VNG
        API_name: Barter.VG | Description: Provides information about Game, DLC, Bundles, Giveaways, Trading
        API_name: IP Geolocation | Description: Geolocate website visitors from their IPs
        API_name: Actinia Grass GIS | Description: Actinia is an open source REST API for geographical data that uses GRASS GIS
        API_name: administrative-divisons-db | Description: Get all administrative divisions of a country
        API_name: adresse.data.gouv.fr | Description: Address database of France, geocoding and reverse
        API_name: BCLaws | Description: Access to the laws of British Columbia
        API_name: Brazil | Description: Community driven API for Brazil Public Data
        API_name: Brazil Receita WS | Description: Consult companies by CNPJ for Brazilian companies
        API_name: Code.gov | Description: The primary platform for Open Source and code sharing for the U.S. Federal Government
        API_name: CMS.gov | Description: Access to the data from the CMS - medicare.gov
        API_name: Coronavirus | Description: HTTP API for Latest Covid-19 Data
        API_name: Covid-19 Datenhub | Description: Maps, datasets, applications and more in the context of COVID-19
        API_name: Covid-19 Government Response | Description: Government measures tracker to fight against the Covid-19 pandemic
        API_name: Adzuna | Description: Job board aggregator
        API_name: Arbeitnow | Description: API for Job board aggregator in Europe / Remote
        API_name: Arbeitsamt | Description: API for the "Arbeitsamt", which is a german Job board aggregator
        API_name: Careerjet | Description: Job search engine
        API_name: AI For Thai | Description: Free Various Thai AI API
        API_name: Cloudmersive | Description: Image captioning, face recognition, NSFW classification
        API_name: Dialogflow | Description: Natural Language Processing
        API_name: EXUDE-API | Description: Used for the primary ways for filtering the stopping, stemming words from the text data
        API_name: 7digital | Description: Api of Music store 7digital
        API_name: AI Mastering | Description: Automated Music Mastering
        API_name: Audiomack | Description: Api of the streaming music hub Audiomack
        API_name: Bandcamp | Description: API of Music store Bandcamp
        API_name: apilayer mediastack | Description: Free, Simple REST API for Live News & Blog Articles
        API_name: Associated Press | Description: Search for news and metadata from Associated Press
        API_name: Chronicling America | Description: Provides access to millions of pages of historic US newspapers from the Library of Congress
        API_name: Currents | Description: Latest news published in various news sources, blogs and forums
        API_name: API Setu | Description: An Indian Government platform that provides a lot of APIS for KYC, business, education & employment
        API_name: Black History Facts | Description: Contribute or search one of the largest black history fact databases on the web
        API_name: BotsArchive | Description: JSON formatted details about Telegram Bots available in database
        API_name: CARTO | Description: Location Information Prediction
        API_name: Countly | Description: Countly web analytics
        API_name: Creative Commons Catalog | Description: Search among openly licensed and public domain works
        API_name: Datamuse | Description: Word-finding query engine
        API_name: Metabase | Description: An open source Business Intelligence server to share data and analytics inside your company
        API_name: EPO | Description: European patent search system api
        API_name: PatentsView | Description: API is intended to explore and visualize trends/patterns across the US innovation landscape
        API_name: TIPO | Description: Taiwan patent search system api
        API_name: USPTO | Description: USA patent api services
        API_name: Advice Slip | Description: Generate random advice slips
        API_name: Biriyani As A Service | Description: Biriyani images placeholder
        API_name: Dev.to | Description: Access Forem articles, users and other resources via API
        API_name: Dictum | Description: API to get access to the collection of the most inspiring expressions of mankind
        API_name: Phone Validation | Description: Validate phone numbers globally
        API_name: apilayer numverify | Description: Phone number validation
        API_name: Cloudmersive Validate | Description: Validate international phone numbers
        API_name: Phone Specification | Description: Rest Api for Phone specifications
        API_name: Bruzu | Description: Image generation with query string
        API_name: CheetahO | Description: Photo optimization and resize
        API_name: Dagpi | Description: Image manipulation and processing
        API_name: Duply | Description: Generate, Edit, Scale and Manage Images and Videos Smarter & Faster
        API_name: Codeforces | Description: Get access to Codeforces data
        API_name: Hackerearth | Description: For compiling and running code in several languages
        API_name: KONTESTS | Description: For upcoming and ongoing competitive coding contests
        API_name: Mintlify | Description: For programmatically generating documentation for code
        API_name: arcsecond.io | Description: Multiple astronomy data sources
        API_name: arXiv | Description: Curated research-sharing platform: physics, mathematics, quantitative finance, and economics
        API_name: CORE | Description: Access the world's Open Access research papers
        API_name: GBIF | Description: Global Biodiversity Information Facility
        API_name: BinaryEdge | Description: Provide access to BinaryEdge 40fy scanning platform
        API_name: BitWarden | Description: Best open-source password manager
        API_name: Botd | Description: Botd is a browser library for JavaScript bot detection
        API_name: Bugcrowd | Description: Bugcrowd API for interacting and tracking the reported issues programmatically
        API_name: Best Buy | Description: Products, Buying Options, Categories, Recommendations, Stores and Commerce
        API_name: Digi-Key | Description: Retrieve price and inventory of electronic components as well as place orders
        API_name: Dummy Products | Description: An api to fetch dummy e-commerce products JSON data with placeholder images
        API_name: eBay | Description: Sell and Buy on eBay
        API_name: 4chan | Description: Simple image-based bulletin board dedicated to a variety of topics
        API_name: Ayrshare | Description: Social media APIs to post, get analytics, and manage multiple users social media accounts
        API_name: aztro | Description: Daily horoscope info for yesterday, today, and tomorrow
        API_name: Blogger | Description: The Blogger APIs allows client applications to view and update Blogger content
        API_name: API-FOOTBALL | Description: Get information about Football Leagues & Cups
        API_name: ApiMedic | Description: ApiMedic offers a medical symptom checker API primarily for patients
        API_name: balldontlie | Description: Balldontlie provides access to stats data from the NBA
        API_name: City Bikes | Description: City Bikes around the world
        API_name: Bacon Ipsum | Description: A Meatier Lorem Ipsum Generator
        API_name: Dicebear Avatars | Description: Generate random pixel-art avatars
        API_name: English Random Words | Description: Generate English Random Words with Pronunciation
        API_name: FakeJSON | Description: Service to generate test and fake data
        API_name: Code Detection API | Description: Detect, label, format and enrich the code in your app or in your data pipeline
        API_name: apilayer languagelayer | Description: Language Detection JSON API supporting 173 languages
        API_name: Aylien Text Analysis | Description: A collection of information retrieval and natural language APIs
        API_name: Detect Language | Description: Detects text language
        API_name: Aftership | Description: API to update, manage and track shipment efficiently
        API_name: Correios | Description: Integration to provide information and prepare shipments using Correio's services
        API_name: Pixela | Description: API for recording and tracking habits or effort, routines
        API_name: PostalPinCode | Description: API for getting Pincode details in India
        API_name: ADS-B Exchange | Description: Access real-time and historical data of any and all airborne aircraft
        API_name: airportsapi | Description: Get name and website-URL for airports by ICAO code
        API_name: AIS Hub | Description: Real-time data of any marine and inland vessel equipped with AIS tracking system
        API_name: Amadeus for Developers | Description: Travel Search - Limited usage
        API_name: CleanURI | Description: URL shortener service
        API_name: ClickMeter | Description: Monitor, compare and optimize your marketing links
        API_name: Drivet URL Shortener | Description: Shorten a long URL easily and fast
        API_name: Free Url Shortener | Description: Free URL Shortener offers a powerful API to interact with other sites
        API_name: Helipaddy sites | Description: Helicopter and passenger drone landing site directory, Helipaddy data and much more
        API_name: Kelley Blue Book | Description: Vehicle info, pricing, configuration, plus much more
        API_name: Mercedes-Benz | Description: Telematics data, remotely access vehicle functions, car configurator, locate service dealers
        API_name: NHTSA | Description: NHTSA Product Information Catalog and Vehicle Listing
        API_name: Catalogopolis | Description: Doctor Who API
        API_name: Catch The Show | Description: REST API for next-episode.net
        API_name: Czech Television | Description: TV programme of Czech TV
        API_name: Dailymotion | Description: Dailymotion Developer API
        API_name: Micro Weather | Description: Real time weather forecasts and historic data
        API_name: ODWeather | Description: Weather and weather webcams
        API_name: Oikolab | Description: 70+ years of global, hourly historical and forecast weather data from NOAA and ECMWF
        API_name: Open-Meteo | Description: Global weather forecast API for non-commercial use
        </apis>

        Please synthesize the corresponding Python function implementations for the API function names and descriptions provided above. The internal logic only needs to be 1 or 2 simple lines.
        Additionally, after synthesizing all the APIs above, you must output the class encapsulation and tools list encapsulation, for example as follows:
            def generate_mock_cc_transaction(customer_id: str, amount: float, merchant_category: str) -> dict:
                # Mock logic: Generate transaction ID "MOCKTX...", category merchant_category
                return {"transaction_id": f"MOCKTX{customer_id}{int(amount)}", "customer_id": customer_id, "amount": amount, "merchant_category": merchant_category}

            def compute_reward_points(customer_id: str, transactions: list[dict]) -> int:
                # Mock logic: reward points = number of transactions * 10
                return len(transactions) * 10      
        For the above two functions’ class wrappers:
            class Tool_generate_mock_cc_transaction(BaseModel):
                customer_id: str = Field(..., description="Identifier of the customer for whom a mock credit card transaction will be generated.")
                amount: float = Field(..., description="The transaction amount used to create mock transaction details and ID.")
                merchant_category: str = Field(..., description="The merchant category associated with the mock transaction.")

            class Tool_compute_reward_points(BaseModel):
                customer_id: str = Field(..., description="Identifier of the customer for whom reward points are being computed.")
                transactions: List[Dict[str, Any]] = Field(..., description="List of transaction records used to compute reward points, each containing transaction details.")
        For the tools list wrapper:
            StructuredTool.from_function(
                func=generate_mock_cc_transaction,
                name="generate_mock_cc_transaction",
                description="Generate a mock credit card transaction record given customer ID, amount, and merchant category.",
                args_schema=Tool_generate_mock_cc_transaction
            ),
            StructuredTool.from_function(
                func=compute_reward_points,
                name="compute_reward_points",
                description="Compute reward points for a customer based on a list of transaction records, awarded at 10 points per transaction.",
                args_schema=Tool_compute_reward_points
            )    
        You must output the complete API code, as well as the corresponding class wrappers and list wrapper, and nothing else.

        '''

    # 创建流式对话
    response_chunks = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model="claude-sonnet-4-5-20250929",  # 指定模型
        max_tokens=49152,
        stream=True  # ➤ 开启流模式
    )

    full_response = ""  # 用于累积完整输出
    # 逐 chunk 读取流式输出
    for chunk in response_chunks:
        # 假设每个 chunk 的结构类似 OpenAI 官方： choices[0].delta.content
        # 你需要根据你客户端库的具体属性名做调整
        delta = chunk.choices[0].delta
        if hasattr(delta, "content") and delta.content:
            text = delta.content
            print(text, end="", flush=True)     # 实时打印输出
            full_response += text                # 累积内容

    # 当流结束后，full_response 即为完整的回复内容
    print("\n\n【流式输出结束】")

    # 将完整回复写入 txt 文件
    with open("response_output.txt", "w", encoding="utf-8") as file:
        file.write(full_response)

    print("完整回复已保存到 response_output.txt 文件")

if __name__ == "__main__":
    main()
