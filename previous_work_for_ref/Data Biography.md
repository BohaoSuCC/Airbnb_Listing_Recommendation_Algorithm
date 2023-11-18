# Data Biography

### Declaration of Authorship

I,jianqiao li, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.

[Jianqian Li]

<br />Date of signature: 24/11/2021
<br />Assessment due date: 26/11/2021
<br />Student Number: 20002335

---
### 1. Who collected the data?

This dataset was created by automatically scraping public information from Airbnb's website and publishing it on Murray Cox's website Inside Airbnb [[1]]((http://insideairbnb.com/about.html)).

---
### 2. Why did they collect it?

Data is collected to quantify the impact of short-term rentals on the Airbnb platform on London's housing and residential communities [[3]]((http://insideairbnb.com/london/neighbourhood=&filterEntireHomes=false&filterHighlyAvailable=false&filterRecentReviews=false&filterMultiListings=false#))(e.g. increased gentrification of communities)[[4]]((http://insideairbnb.com/london/neighbourhood=&filterEntireHomes=false&filterHighlyAvailable=false&filterRecentReviews=false&filterMultiListings=false#)). The data will not be subject to the commercial interests of Airbnb, which allows for public and government discussion and analysis, and protects community development and balance[[2]]((http://insideairbnb.com/behind.html)) .

---
### 3. How was it collected?

The data is gathered from the Airbnb website using crawler technology (python), then stored in a database (PostgreSQL) as csv files. The data is then validated, cleaned, analyzed, and aggregated before being published on the website. Certain open-source technologies (D3, Cross filter, dc.js, Leaflet, and so on) and some incredible Airbnb code from Tom Slee contribute to the accuracy and readability of the data[[2]]((http://insideairbnb.com/behind.html)). Additionally, occupancy models were used to measure the effects of Airbnb on housing.

---
### 4. What useful information does it contain?

Through analysis, the 74 columns of data in the database can be divided into four categories useful information.
1. Spatial location information: latitude and longitude information and neighborhood information, which is used to support spatial analysis.
2. Host information: id, name, location, photo, total number of listings, etc. This type of information is used to obtain a user profile of the Host. In addition, host about data(textual) can support feature analysis.
3. Listing information: type, price, facilities, housing license, days of access, etc., with spatial data to identify the property information of each house, as well as the legitimacy of the rental properties.
4. Review information: quantity, cleanliness, communication, convenience of location, etc. to give a quantitative evaluation analysis of each property.

---
### 5. To what extent is the data 'complete'?

The data is incomplete, and according to the Q4 summary, there is adequate information to answer basic inquiries about Airbnb in any area or throughout the city. Through the visualizations of geographical data of listings, we may also use data to address more complicated issues, such as whether short-term rentals are contributing to Gentrification and geographically uneven[[4]]((http://insideairbnb.com/behind.html)).

However, there are some defects and incompleteness because there is some structurally missing data. The data cannot give solutions to concerns such as the impact of quantifying the rental market in high-density urban zones, such as the London area, due to its inaccuracy in terms of geographical data and the lack of temporal consistency of rental information.  Furthermore, when we utilize the data to investigate racism in the Airbnb platform[[5]]((https://journals-sagepub-com.libproxy.ucl.ac.uk/doi/10.1177/0308518X19886321?icid=int.sj-full-text.similar-articles.2)), the lack of information about hosts and tenants' race significantly detracts from the overall 'picture' of our research. Overall, the data is raw, and it needs to be cleansed (by removing nulls and unnecessary information) and transformed (textual information and price information) to suit our research concerns better.

---
### 6. What kinds of analysis would this support?
The dataset supports two types of data analysis: fundamental one-dimensional data analysis and advanced multi-dimensional data analysis.
1. **One-dimensional data analysis**:
- Spatial data analysis: isualizing and mapping the location information and neighborhoods associated with the   listings referenced in question 4 to determine the quantity, concentration, and spatial significance of Airbnb listings in a region[[6]]((http://insideairbnb.com/reports/how-airbnbs-data-hid-the-facts-in-new-york-city.pdf.)).
- Numeric Data Analysis: This type of analysis allows for some quantitative description of the data, for example, using regression analysis to explore the existence of a correlation between price and the rating of a property.
- Textual Data Analysis: The text analysis can be aided by the data columns 'description,' 'host about,' and 'host response time.' A cluster analysis of the listing descriptions may determine the most frequently used keywords and create a word cloud.

2. **Advanced multi-dimensional data analysis**:
- The Socio-spatial Analysis: We mix data from different dimensions, because a single dimension of data is insufficient to investigate socio-spatial issues. For example, when we assess the socio-spatial outcomes of short-term rentals[[7]]((https://journals-sagepub-com.libproxy.ucl.ac.uk/doi/full/10.1177/0308518X19869012)), it is necessary to combine the spatial data with numeric data (price, availability day) in the analysis process. In other words, using multi-dimensional analysis would better determine how do the Airbnb market reshapes the community's society and space.

---
### 7. Which of the uses presented in Q.6 are ethical?
The uses mentioned in Q.6 are ethical.

Firstly, there is always a risk of privacy exposure with any data, resulting in ethical debate. As a data analyst, it is essential to strike **_a balance between security and privacy_**[[8]]((https://www-tandfonline-com.libproxy.ucl.ac.uk/doi/full/10.1080/03098265.2018.1436534?scroll=top&needAccess=true)), rather than to tell people that the data does not pose any risk of privacy breach, which is unethical. The  Airbnb website provides data from the dataset, and users on the site voluntarily share all data. In addition, Airbnb anonymizes the location information, which protects the security of the listings while not having a significant impact on the results of the spatial location study in Q.6. Then, when we aggregate some data, we need to be careful with the analysis. In the case of host information, even if all the information is publicly available when processing the data, data analysts can infer the gender, age, ethnicity of the landlord from their photo, name, address, and ownership data, which is an ethical violation in data analysis. Therefore, we need to be aware of privacy issues at any stage involving data, not just cleaning up information that might reveal privacy at the time of data collection.

Secondly, **_context_** is essential for ethical analysis[9], as the data are not neutral or objective; they result from unequal social relations. Inside Airbnb explains the meaning of each column in the dataset. Moreover, Murray Cox also explains the dataset's aim to achieve - "how Airbnb is being used to compete with the residential housing market.[[2]]((http://insideairbnb.com/behind.html))"The importance of context is not only in the data collection and analysis phase but also in interpreting and communicating the results. The result of all the uses of Q.6 is not cold numbers or cool-looking maps, but an analysis that explores the spatial-social outcomes hidden in the Airbnb platform and London and its impact on people. The bottom line for data is that they cannot speak for themselves because data derive from a data set influenced by power differentials. It is, therefore, the responsibility of the data analyst to prevent the data from speaking for itself, which is one of the fundamental ethics of data.

---
## Bibliography
[1]&emsp;M.Cox.(2016). _About Inside Airbnb_ [Online] . Available: http://insideairbnb.com/about.html

[2]&emsp;M.Cox.(2016). _Behind Inside Airbnb_ [Online] . Available: http://insideairbnb.com/behind.html

[3]&emsp;M.Cox.(2016)._Airbnb in London_.[Online] . Available:                               http://insideairbnb.com/london/neighbourhood=&filterEntireHomes=false&filterHighlyAvailable=false&filterRecentReviews=false&filterMultiListings=false#

[4]&emsp;D.Wachsmuth,A.Weisler.(2018,June) "Airbnb and the rent gap: Gentrification through the sharing economy".Environment and Planning A: Economy and Space, 50(6), pp. 1147–1170.Available:https://journals-sagepub-com.libproxy.ucl.ac.uk/doi/10.1177/0308518X18778038

[5]&emsp;P.Törnberg , L.Chiappini."Selling black places on Airbnb: Colonial discourse and the marketing of black communities in New York City".Environment and Planning A: Economy and Space, 52(3), pp. 553–572.Available:https://journals-sagepub-com.libproxy.ucl.ac.uk/doi/10.1177/0308518X19886321?icid=int.sj-full-text.similar-articles.2

[6]&emsp;M. Cox and T. Slee.(2016,Fed) “How Airbnb’s Data Hid the Facts in New York City.” Inside Airbnb [online], Available: http://insideairbnb.com/reports/how-airbnbs-data-hid-the-facts-in-new-york-city.pdf.

[7]&emsp;A.Cocola-Gant, A.Gago.(2019,August)"Airbnb, buy-to-let investment and tourism-driven displacement": A case study in Lisbon.Environment and Planning A: Economy and Space, 53(7), pp. 1671–1688.Available:https://journals-sagepub-com.libproxy.ucl.ac.uk/doi/full/10.1177/0308518X19869012

[8]&emsp;V.Bemt,J.Doornbos,L.Meijering,M.Plegt, et al.(2019,August)"Teaching ethics when working with geocoded data: A novel experiential learning approach.",Journal of Geography in Higher Education", 53(7), pp. 1671–1688.Available:https://www-tandfonline-com.libproxy.ucl.ac.uk/doi/full/10.1080/03098265.2018.1436534?scroll=top&needAccess=true

[9]&emsp;C.D'Ignaziov and L.F.Klein,"The Numbers Don't Speak for Themselves," in Data feminism, D.Weinberger, Ed.The MIT Press,2020,pp.149-170
