library(shiny)
library(DT)
library(fresh)
library(bs4Dash)
library(fullPage)
library(shinybulma)
library(shinyEffects)
library(keras)
library(billboarder)
library(tidyverse)
library(tidyquant)
library(corrr)
library(scales)
library(lime)
library(glue)
library(rsample)
library(recipes)
library(yardstick)
library(readr)
library(ggplot2)
library(forcats)
telco <- readxl::read_excel("D:/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.xlsx")
load('D:/R/Churn Prediction/customer_churn.RData')
model_keras <- load_model_hdf5('D:/R/Churn Prediction/customer_churn.hdf5', compile = FALSE)
main_vars <- c('tenure', 'Contract', 'InternetService', 'MonthlyCharges', 
               'OnlineBackup', 'OnlineSecurity', 'DeviceProtection', 
               'TechSupport', 'StreamingMovies', 'PhoneService')
commercial_vars <- c('InternetService', 'OnlineBackup', 'OnlineSecurity', 
                     'DeviceProtection', 'TechSupport', 'StreamingMovies', 
                     'PhoneService')
financial_vars <- c('PaymentMethod')
customer_feature_vars <- c(main_vars, commercial_vars, financial_vars) %>% unique
churn_data_raw <-  readxl::read_excel("D:/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.xlsx") %>% 
  mutate(
    tenure_range = case_when(
      tenure < 12 ~ '< 1 Yr',
      tenure < 24 ~ '1-2 Yrs',
      tenure < 36 ~ '2-3 Yrs',
      tenure >= 36 ~ 'Over 3 Yrs',
      TRUE ~ 'NA'
    ),
    monthly_charge_range = case_when(
      MonthlyCharges < 20 ~ '< 20 per Month',
      MonthlyCharges < 50 ~ '20-50 per Month',
      MonthlyCharges < 100 ~ '50-100 per Month',
      MonthlyCharges >= 100 ~ 'Over 100 per Month',
      TRUE ~ 'NA'
    )
  )
churn_data_tbl <- churn_data_raw %>%
  drop_na() %>%
  select(Churn, everything())

assign("model_type.keras.engine.sequential.Sequential", envir = globalenv(), function(x, ...) {
  "classification"
})

assign("predict_model.keras.engine.sequential.Sequential", envir = globalenv(), function(x, newdata, type, ...) {
  pred <- predict_proba(object = x, x = as.matrix(newdata))
  data.frame(Yes = pred, No = 1 - pred)
})



options <- list(
  sectionsColor = c("#424951","#424951","#424951","#424951","#424951","#424951"),
  parallax = TRUE
)
ui <- dashboardPage(dark = TRUE, freshTheme = create_theme(

  bs4dash_status(
   danger = "#6fc476", warning  = "#ffd764", success = "#725ae4", info = "#f20000"
  )
),
  dashboardHeader(
                  skin = "dark",
                  status = "primary",
                  leftUi = NULL,
                  rightUi = tagList(
                    dropdownMenu(
                      badgeStatus = "primary",
                      type = "notifications",
                      notificationItem(
                        userMessages(
                          width = 12,
                          status = "danger",
                          userMessage(
                            author = "AleX Paul",
                            date = "20 June 2:00 pm",
                            image = "https://static.vecteezy.com/system/resources/previews/000/142/007/original/headshot-of-smiling-young-man-with-beard-vector.jpg",
                            type = "received",
                            "Hi"
                          ),
                          userMessage(
                            author = "Merin ",
                            date = "23 June 2:05 pm",
                            image = "https://static.vecteezy.com/system/resources/previews/000/141/702/original/headshot-of-smiling-beautiful-employee-vector.jpg",
                            type = "sent",
                            "Hello"
                          )
                        )
                      )
                    ),
                    dropdownMenu(
                      badgeStatus = "info",
                      type = "tasks",
                      taskItem(
                        inputId = "triggerAction3",
                        text = "My progress",
                        color = "orange",
                        value = 70
                      ),
                      taskItem(
                        inputId = "triggerAction3",
                        text = "My progress",
                        color = "indigo",
                        value = 85
                      )
                    )
                  )),
  dashboardSidebar( 
    sidebarMenu(
      id = "sidebarMenu",
     
      bs4SidebarUserPanel(name="Customer-Churn", image = "https://webstockreview.net/images/employee-clipart-call-center-16.png"),
      box(
        width = NULL,
        h2(icon("tty"),"Customer."),
        h3("Churn-Analysis"),
        closable = FALSE,
        collapsible = TRUE,
        elevation = 4,
        
        ""
      ),
     
      bs4SidebarHeader(title = "Welcome to DashBoard"),
      menuItem(
        text = "Team:Insight's",
        icon = icon("users"),
        startExpanded = FALSE,
        badgeColor = "success",
        menuSubItem(
          text = "Merin George(20BDA11)",
          tabName = "link1",
          icon = icon("	fas fa-grin-beam")
        ),
        menuSubItem(
          text = "Ranjith Kumar(20BDA56)",
          tabName = "tab4",
          icon = icon("	fas fa-dizzy")
        ),
        menuSubItem(
          text = "Alex Paul(20BDA66)",
          tabName = "tab4",
          icon = icon("fas fa-grimace")
        )
      ),
      menuItem(
        text = "Customer ID",
        tabName = "link2",
        icon = icon("id-badge"),
        startExpanded = FALSE,
        badgeColor = "success",
        bs4Card(width = 12,id = "my-progress",
                background = "gray-dark",
        selectInput('customer_id', NULL, unique(test_tbl_with_ids$customerID)))
      ),
      menuItem(
        condition = "input.show == true",
        text = "Churn Facets",
        tabName = "link2",
        icon = icon("address-card"),
        startExpanded = FALSE,
        badgeColor = "primary",

        bs4Card(width = 12,id = "my-progress",
                background = "gray-dark",
        selectInput('payment_methods', 'Payment Method', 
                    c('All', unique(churn_data_raw$PaymentMethod))),
        selectInput('tech_support', 'Tech Support', 
                    c('All', unique(churn_data_raw$TechSupport))),
        selectInput('monthly_charge_range', 'Monthly Charge Range', 
                    c('All', unique(churn_data_raw$monthly_charge_range))),
        selectInput('tenure_range', 'Tenure Range', 
                    c('All', unique(churn_data_raw$tenure_range))))
        
        
      )
    )), controlbar = dashboardControlbar(),
  dashboardBody( setZoom(class = "box"),
                 setZoom(id = "my-progress"),fullPage(
    menu = c("Full Page" = "link1",
             "Sections" = "link2",
             "Slides" = "section3",
             "backgrounds" = "section4",
             "Background Slides" = "section5",
             "Callbacks" = "section6"),
    opts = options,
    fullSectionImage(  jumbotron(
      title = "Customer Churn Analysis",
      lead = "Customer churn occurs when customers or subscribers stop doing business with a company or service, also known as customer attrition. It is also referred as loss of clients or customers. One industry in which churn rates are particularly useful is the telecommunications industry, because most customers have multiple options to choose  location.",
      "GitHUB info",
      status = "primary",
      href = "https://github.com/Rangow4562/Customer-Churn-Analysis"
    ),
      center = TRUE,
    #height = "500px",
      img = "https://www.linkpicture.com/q/3_945.jpg",height = "500px",
      menu = "link1",
      tags$h1("")
    ),
    fullSection(
      menu = "link2",box(
                         title = "Customer-Churn DataView",
                         closable = TRUE,
                         background = "white",
                         elevation = 4,
                         width = 12,
                         status = "success",
                         gradient = TRUE,
                         solidHeader = TRUE,
                         collapsible = TRUE,
      fullContainer(
        fullRow(
          fullColumn(
            width=12,
            DT::dataTableOutput("AirPassengers",height = "650px")
          ),
        
        )
      ))
    ),
    fullSection(
      menu = "section3",
      fullSlide( 
        fullContainer(
          center = TRUE,   fluidRow(
            bs4ValueBoxOutput("main"),
            bs4ValueBoxOutput("commercial"),
            bs4ValueBoxOutput("financial")
          ),
          fullRow( box(id = "my-progress",
                       title = "Customer-Churn DataView",
                       closable = TRUE,
                       background = "white",
                       elevation = 4,
                       width = 4,
                       status = "danger",
                       gradient = TRUE,
                       solidHeader = TRUE,
                       collapsible = TRUE,
                       
          DT::dataTableOutput('customer_info_tbl',height = "650px")),
          
          box(id = "my-progress",
              title = "Contributions to Churn(LIME)",
              closable = TRUE,
              width = 8,
              background = "white",
              elevation = 4,
              status = "danger",
              gradient = TRUE,
              maximizable = TRUE,
              solidHeader = TRUE,
              collapsible = TRUE,
              billboarderOutput('customer_explanation',height = "650px"))),
          shiny::verbatimTextOutput("containerCode")
        )
      ),
      fullSlide(fullContainer(
        center = TRUE,
        fullRow(box(id = "my-progress",
                    title = "Monthly revenue by type of contract",
                    closable = TRUE,
                    background = "white",
                    width = 6,
                    status = "success",
                    elevation = 4,
                    gradient = TRUE,
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    
                    billboarderOutput('monthly_revenue',height = "150px")),
                box(id = "my-progress",
                    title = "Number of customers by type of contract",
                    closable = TRUE,
                    background = "white",
                    width = 6,
                    status = "success",
                    elevation = 4,
                    gradient = TRUE,
                    solidHeader = TRUE,
                    collapsible = TRUE,
                    
                    billboarderOutput('number_of_customers',height = "150px")),
                
                ),
        
        fluidRow(
          box(id = "my-progress",
              title = "Monthly revenue churn",
              closable = TRUE,
              background = "white",
              width = 6,
              status = "success",
              elevation = 4,
              gradient = TRUE,
              solidHeader = TRUE,
              collapsible = TRUE,
              
              billboarderOutput('pct_monthly_revenue',height = "150px")),
          box(id = "my-progress",
              title = "Customer churn",
              closable = TRUE,
              background = "white",
              width = 6,
              status = "success",
              elevation = 4,
              gradient = TRUE,
              solidHeader = TRUE,
              collapsible = TRUE,
              
              billboarderOutput('pct_customers',height = "150px"))
          
        ),
        fluidRow(
          box(id = "my-progress",
              title = "Churn rate by tenure range",
              closable = TRUE,
              background = "white",
              width = 6,
              status = "success",
              gradient = TRUE,
              solidHeader = TRUE,
              elevation = 4,
              collapsible = TRUE,
              billboarderOutput('churn_rate_tenure',height = "150px")),
          box(id = "my-progress",
              title = "Churn rate by internet service ",
              closable = TRUE,
              background = "white",
              width = 6,
              status = "success",
              gradient = TRUE,
              solidHeader = TRUE,
              elevation = 4,
              collapsible = TRUE,
              billboarderOutput('churn_rate_internet_service',height = "150px")
              )
          
          
          
        )
        
      )
       
      ),fullSlide(
                          fullContainer(fluidRow(
        box(id = "my-progress",
            title = "Churn rate Correlation Analysis ",
            closable = TRUE,
            background = "white",
            width = 12,
            status = "warning",
            gradient = TRUE,
            solidHeader = TRUE,
            elevation = 4,
            collapsible = TRUE,
            
            plotOutput('corr_analysis',height = "700px"))
        
        
        
        
      )))
    ),

    
    fullSection(
      menu = "section6",
      center = TRUE,
      h2("Team: Insights"),
      fluidRow(  userBox(
        title = userDescription(
          title = "",
          subtitle = "Merin George(20BDA11) - R project Team Leader",
          image = "https://www.linkpicture.com/q/Untitled-1_39.jpg",
          backgroundImage = "https://images.unsplash.com/photo-1598350742412-8fe67cd5375b?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80"
        ),
        status = "indigo",
        closable = TRUE,
        elevation = 4,
        footer = ""
      ),
      userBox(
        title = userDescription(
          title = "",
          subtitle = "Nagavarun(20BDA56) - Python project Team Leader",
          image = "https://www.linkpicture.com/q/1621181683969.jpg",
          backgroundImage = "https://images.unsplash.com/photo-1606978299204-fe44c0b3ee10?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80",
        ),
        status = "olive",
        closable = TRUE,
        elevation = 4,
        footer = ""
      ),
      ),
      fluidRow(  userBox(
        title = userDescription(
          title = "",
          subtitle = "Alex Paul(20BDA66) - R Project Team",
          type = 1,
          image = "https://www.linkpicture.com/q/Untitled-2_19.jpg",
          backgroundImage = "https://images.unsplash.com/photo-1585082928729-2cf5033f3b60?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1189&q=80"
        ),
        status = "indigo",
        closable = TRUE,
        elevation = 4,
        footer = ""
      ),
      userBox(
        title = userDescription(
          title = "",
          subtitle = "Rakshith Kumar(20BDA47) - Python Project Team",
          image = "https://www.linkpicture.com/q/Untitled-3_22.jpg",
          backgroundImage = "https://images.unsplash.com/photo-1606978302477-86dbf8a8bdef?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80",
        ),
        status = "olive",
        closable = TRUE,
        elevation = 4,
        footer = ""
      )),
      fluidRow(  userBox(
        title = userDescription(
          title = "",
          subtitle = "Ranjith Kumar(20BDA56) - R Project Team",
          type = 1,
          image = "https://www.linkpicture.com/q/Untitled-4_14.jpg",
          backgroundImage = "https://images.unsplash.com/photo-1531317994335-9222558fa07a?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=858&q=80"
        ),
        status = "indigo",
        closable = TRUE,
        elevation = 4,
        footer = ""
      ),
      )
     
    )
  )))

server <- function(input, output) {
  churn_analysis_data <- reactive({
    
    churn_data_filtered <- churn_data_raw
    
    if (input$payment_methods != 'All') {
      churn_data_filtered <- filter(churn_data_filtered, PaymentMethod == input$payment_methods)
    }
    
    if (input$tech_support != 'All') {
      churn_data_filtered <- filter(churn_data_filtered, TechSupport == input$tech_support)
    }
    
    if (input$monthly_charge_range != 'All') {
      churn_data_filtered <- filter(churn_data_filtered, monthly_charge_range == input$monthly_charge_range)
    }
    
    if (input$tenure_range != 'All') {
      churn_data_filtered <- filter(churn_data_filtered, tenure_range == input$tenure_range)
    }
    
    churn_data_filtered
  })
  bb_colors <- function(bb) {
    bb %>% bb_colors_manual('Yes' = 'rgba(7, 0, 125, 0.66)', 'No' = 'rgba(221, 22, 0, 0.55)')
  }

  output$customer_info_tbl <- DT::renderDataTable({
    
    req(input$customer_id)
    
    selected_customer_id <- test_tbl_with_ids$customerID[1]
    selected_customer_id <- input$customer_id
    
    customer_info <- test_tbl_with_ids %>% 
      filter(customerID == selected_customer_id) %>% 
      mutate(tenure = paste0(tenure, ifelse(tenure == 1, ' Month', ' Months'))) %>% 
      select(customer_feature_vars) %>% 
      gather(metric, value)
    
    DT::datatable(
      customer_info, 
      rownames = NULL, 
      options = list(
        dom = 't', 
        bSort = TRUE, 
        paging = TRUE
      )
    )
  })
  observeEvent(input$strategy_box_hover, {
    
    strategy_hover <- input$strategy_box_hover
    
    if (strategy_hover == 'none') {
      row_indices <- 0
    } else {
      strategy_features <- get(paste0(strategy_hover, '_vars'))
      row_indices <- match(strategy_features, customer_feature_vars)
    }
    
    DT::dataTableProxy('customer_info_tbl') %>% 
      DT::selectRows(row_indices)
  })
  

  output$AirPassengers <- DT::renderDataTable({
    DT::datatable(churn_data_raw,
                  extensions = 'FixedHeader',
                  options = list(
                    fixedHeader = TRUE,
                    scrollX = TRUE
                  )
    )
  })


  
  output$customer_explanation <- renderBillboarder({
    
    req(input$customer_id)
    
    selected_customer_id <- test_tbl_with_ids$customerID[1]
    selected_customer_id <- input$customer_id
    
    # Run lime() on training set
    explainer <- lime::lime(
      x = x_train_tbl,
      model = model_keras,
      bin_continuous = FALSE
    )
    
    customer_index <- test_tbl_with_ids %>% 
      mutate(rownum = row_number()) %>% 
      filter(customerID == selected_customer_id) %>%
      select(rownum)
    
    # Run explain() on explainer
    set.seed(42)
    explanation <- explain(
      x_test_tbl[customer_index$rownum,], 
      explainer = explainer, 
      n_labels = 1, 
      n_features = length(x_test_tbl),
      kernel_width = 0.5
    )
    
    type_pal <- c('Supports', 'Contradicts')
    explanation$type <- factor(ifelse(sign(explanation$feature_weight) == 
                                        1, type_pal[1], type_pal[2]), levels = type_pal)
    description <- paste0(explanation$case, "_", explanation$label)
    desc_width <- max(nchar(description)) + 1
    description <- paste0(format(description, width = desc_width), 
                          explanation$feature_desc)
    explanation$description <- factor(description, levels = description[order(abs(explanation$feature_weight))])
    explanation$case <- factor(explanation$case, unique(explanation$case))
    
    explanation_plot_df <- explanation %>%
      mutate(churn_predictor = case_when(
        (label == 'Yes' & type == 'Supports') | (label == 'No' & type == 'Contradicts') ~ 'More likely to churn',
        (label == 'Yes' & type == 'Contradicts') | (label == 'No' & type == 'Supports') ~ 'Less likely to churn'
      )) %>%
      arrange(-abs(feature_weight)) %>% 
      head(20)
    
    billboarder() %>%
      bb_barchart(
        data = explanation_plot_df,
        mapping = bbaes(x = feature_desc, y = feature_weight, group = churn_predictor),
        rotated = TRUE,
        stacked = TRUE
      ) %>%
      bb_colors_manual('Less likely to churn' = 'rgba(7, 0, 125, 0.66)', 'More likely to churn' = 'rgba(221, 22, 0, 0.55)')
  })
  set.seed(122)
  histdata <- rnorm(500)
  
  output$plot1 <- renderPlot({
    data <- histdata[seq_len(input$slider)]
    hist(data)
  })
  
  output$monthly_revenue <- renderBillboarder({
    
    plot_df <- churn_analysis_data() %>% 
      group_by(Churn, Contract) %>% 
      summarise(monthly_revenue = sum(MonthlyCharges))
    
    billboarder() %>% 
      bb_barchart(
        data = plot_df,
        mapping = bbaes(x = Contract, y = monthly_revenue / 10000, group = Churn),
        stacked = TRUE,
        rotated = TRUE
      ) %>% 
      bb_y_axis(label = list(text = "Revenue (USD, in thousands)",
                             position = "outer-top")) %>% 
      bb_colors()
  })
  output$number_of_customers <- renderBillboarder({
    
    plot_df <- churn_analysis_data() %>% 
      group_by(Churn, Contract) %>% 
      summarise(number_of_customers = n())
    
    billboarder() %>% 
      bb_barchart(
        data = plot_df,
        mapping = bbaes(x = Contract, y = number_of_customers, group = Churn),
        stacked = TRUE,
        rotated = TRUE
      ) %>% 
      bb_y_axis(label = list(text = "Customer Count",
                             position = "outer-top")) %>%
      bb_colors()
  })
  output$pct_monthly_revenue <- renderBillboarder({
    
    plot_df <- isolate(churn_analysis_data()) %>% 
      group_by(Churn) %>% 
      summarise(monthly_revenue = sum(MonthlyCharges)) %>% 
      ungroup %>% 
      mutate(pct = round(monthly_revenue / sum(monthly_revenue), 2)) %>% 
      select(-monthly_revenue) %>% 
      mutate(x = 'Churn') %>% 
      spread(Churn, pct)
    
    billboarder() %>% 
      bb_barchart(
        data = plot_df,
        stacked = TRUE,
        rotated = TRUE
      ) %>% 
      bb_y_axis(label = list(text = "Percentage, Monthly Revenue",
                             position = "outer-top")) %>%
      bb_colors()
  })
  output$pct_customers <- renderBillboarder({
    
    plot_df <- churn_analysis_data() %>% 
      group_by(Churn) %>% 
      summarise(num_customers = n()) %>% 
      ungroup %>% 
      mutate(pct = round(num_customers / sum(num_customers), 2)) %>% 
      select(-num_customers) %>% 
      mutate(x = 'Churn') %>% 
      spread(Churn, pct)
    
    billboarder() %>% 
      bb_barchart(
        data = plot_df,
        stacked = TRUE,
        rotated = TRUE
      ) %>% 
      bb_y_axis(label = list(text = "Percentage, Customers",
                             position = "outer-top")) %>%
      bb_colors()
  })
  output$churn_rate_tenure <- renderBillboarder({
    
    plot_df <- churn_analysis_data() %>% 
      count(tenure_range, Churn) %>% 
      group_by(tenure_range) %>% 
      mutate(pct = round(n / sum(n), 2)) %>% 
      ungroup
    
    plot <- billboarder() %>% 
      bb_y_grid(
        lines = list(
          list(value = mean(churn_analysis_data()$Churn == 'Yes'), text = "Average Churn Rate")
        )
      ) %>% 
      bb_y_axis(label = list(text = "Percentage, Customers",
                             position = "outer-top")) %>%
      bb_colors()
    
    if (nrow(plot_df) == 2) {
      plot_df <- plot_df %>% 
        select(-n) %>% 
        spread(Churn, pct)
      
      plot <- plot %>% 
        bb_barchart(
          data = plot_df,
          stacked = TRUE,
          rotated = TRUE
        )
    } else {
      plot <- plot %>% 
        bb_barchart(
          data = plot_df,
          mapping = bbaes(x = tenure_range, y = pct, group = Churn),
          stacked = TRUE,
          rotated = TRUE
        ) 
    }
    
    plot
  })
  
  output$churn_rate_internet_service <- renderBillboarder({
    plot_df <- churn_analysis_data() %>% 
      count(InternetService, Churn) %>% 
      group_by(InternetService) %>% 
      mutate(pct = round(n / sum(n), 2))
    
    billboarder() %>% 
      bb_barchart(
        data = plot_df,
        mapping = bbaes(x = InternetService, y = pct, group = Churn),
        stacked = TRUE,
        rotated = TRUE
      ) %>% 
      bb_y_grid(
        lines = list(
          list(value = mean(churn_analysis_data()$Churn == 'Yes'), text = "Average Churn Rate")
        )
      ) %>% 
      bb_y_axis(label = list(max = 1, 
                             text = "Percentage, Customers",
                             position = "outer-top")) %>%
      bb_colors()
  })
  
  output$churn_data_tbl = DT::renderDataTable({
    churn_data_tbl
  })

  
  
  output$corr_analysis <- renderPlot({
    
    withProgress(message = 'Generating correlations plot', value = 0.6, {
      
     
      corrr_analysis <- x_train_tbl %>%
        as_tibble() %>%
        mutate(Churn = y_train_vec) %>%
        correlate() %>%
        focus(Churn) %>%
        rename(feature = term) %>%
        arrange(abs(Churn)) %>%
        mutate(feature = as_factor(feature)) 
      
     
      corrr_analysis %>%
        ggplot(aes(x = Churn, y = fct_reorder(feature, desc(Churn)))) +
        geom_point() +
        # Positive Correlations - Contribute to churn
        geom_segment(aes(xend = 0, yend = feature),
                     color = palette_light()[[2]],
                     data = corrr_analysis %>% filter(Churn > 0)) +
        geom_point(color = palette_light()[[2]],
                   data = corrr_analysis %>% filter(Churn > 0)) +
        # Negative Correlations - Prevent churn
        geom_segment(aes(xend = 0, yend = feature),
                     color = palette_light()[[1]],
                     data = corrr_analysis %>% filter(Churn < 0)) +
        geom_point(color = palette_light()[[1]],
                   data = corrr_analysis %>% filter(Churn < 0)) +
        # Vertical lines
        geom_vline(xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
        geom_vline(xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
        geom_vline(xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
        # Aesthetics
        theme_tq() +
        labs(title = "Churn Correlation Analysis",
             subtitle = "Positive Correlations (contribute to churn), Negative Correlations (prevent churn)",
             y = "Feature Importance")
    })
  })
  
  output$main <- renderbs4ValueBox({
    
    req(input$customer_id)
    
    selected_customer_id <- test_tbl_with_ids$customerID[1]
    selected_customer_id <- input$customer_id
    
    customer_tbl <- test_tbl_with_ids %>% 
      filter(customerID == selected_customer_id)
    
    if (customer_tbl$tenure <= 9) {
      main_strategy <- 'Retain until one year'
    } else if (customer_tbl$tenure > 9 | customer_tbl$Contract == 'Month-to-month') {
      main_strategy <- 'Upsell to annual contract'
    } else if (customer_tbl$tenure > 12 & customer_tbl$InternetService == 'No') {
      main_strategy <- 'Offer internet service'
    } else if (customer_tbl$tenure > 18 & customer_tbl$MonthlyCharges > 50) {
      main_strategy <- 'Offer discount in monthly rate'
    } else if (customer_tbl$tenure > 12 & 
               customer_tbl$Contract != 'Month-to-month' & 
               ((customer_tbl$OnlineBackup == 'No' & 
                 customer_tbl$OnlineSecurity == 'No' & 
                 customer_tbl$DeviceProtection == 'No' & 
                 customer_tbl$TechSupport == 'No' & 
                 customer_tbl$StreamingMovies == 'No') 
                | customer_tbl$PhoneService == 'No')) {
      main_strategy <- 'Offer additional services'
    } else {
      main_strategy <- 'Retain and maintain'
    }
    bs4ValueBox(subtitle = "", value = main_strategy,gradient = TRUE,    elevation = 4,color = "danger",icon = icon("calendar-times"),footer = div("Main Strategy"))
  })
  output$commercial <- renderValueBox({
    
    req(input$customer_id)
    
    selected_customer_id <- test_tbl_with_ids$customerID[1]
    selected_customer_id <- input$customer_id
    
    customer_tbl <- test_tbl_with_ids %>% 
      filter(customerID == selected_customer_id)
    
    if ((customer_tbl$InternetService == 'DSL' & 
         customer_tbl$OnlineBackup == 'No' & 
         customer_tbl$OnlineSecurity == 'No' & 
         customer_tbl$DeviceProtection == 'No' & 
         customer_tbl$TechSupport == 'No' & 
         customer_tbl$StreamingMovies == 'No') 
        | customer_tbl$PhoneService == 'No') {
      commercial_strategy <- 'Offer additional services'
    } else if (customer_tbl$InternetService == 'Fiber optic') {
      commercial_strategy <- 'Offer tech support and services'
    } else if (customer_tbl$InternetService == 'No') {
      commercial_strategy <- 'Upsell to internet service'
    } else {
      commercial_strategy <- 'Retain and maintain'
    }
    
    bs4ValueBox(subtitle ="", value = commercial_strategy, gradient = TRUE,   elevation = 4,color = "success",icon = icon("file-alt"),footer = div("Commercial Strategy"))
  })
  
  output$financial <- renderValueBox({
    
    req(input$customer_id)
    
    selected_customer_id <- test_tbl_with_ids$customerID[1]
    selected_customer_id <- input$customer_id
    
    customer_tbl <- test_tbl_with_ids %>% 
      filter(customerID == selected_customer_id)
    
    if (customer_tbl$PaymentMethod %in% c('Mailed Check', 'Electronic Check')) {
      financial_strategy <- 'Move to credit card or bank transfer'
    } else {
      financial_strategy <- 'Retain and maintain'
    }
    bs4ValueBox(subtitle ="", value = financial_strategy,gradient = TRUE,   elevation = 4, color = "warning",icon = icon("google-wallet"),footer = div("Financial Strategy"))
  
  })
  
}

shinyApp(ui, server)