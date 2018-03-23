  library("ggalluvial")
  library("rmarkdown")
  library("rJava")
  library("xlsxjars" )
  library("xlsx" )
  library("XLConnectJars" )
  library("XLConnect" )
  library("WriteXLS" )
  library("tidyr" )
  library("tibble" )
  library("stringr" )
  library("stringi" )
  library("rJava" )
  library("readr" )
  library("R6" )
  library("Rcpp" )
  library("magrittr" )
  library("lazyeval" )
  library("lubridate" )
  library("jsonlite" )
  library("hms" )
  library("gtools" )
  library("gdata" )
  library("dplyr" )
  library("DBI" )
  library("curl" )
  library("BH" )
  library("assertthat" )
  library("readxl" )
  library("ggplot2" )
  library(darksky)
  library(xts)
  library(sp)
  library(rgdal)
  library(geosphere)
  library(scan)
  library(bigmemory)
  library(biganalytics)
  library(bigtabulate)
  library(RANN)
  library(data.table)
  library(lme4)
  library(gridExtra)
  library(gmodels)
  library(readr)
  library(gmodels)
  library(GGally)
  library(stargazer)
  library(pscl)
  #library(Rz)
  library(car)
  library(scales)
  
  #Introduzco las constantes y la  matriz de traduccion
  file<-c("Austria","Bulgaria","Croatia","Cyprus","Czech Republic","Denmark","England","Estonia","Finland","France","Germany","Greece","Hungary","Ireland","Italy","Lithuania","Netherlands","Northern Ireland","Poland","Portugal","Romania","Scotland","Slovakia","Slovenia","Spain","Sweden","Wales")
  insideFile<-c("at","bg","hr","cy","cz","dk","eng","ee","fi","fr","de","gr","hu","ie","it","lt","nl","nir","pl","pt","ro","sct","sk","si","es","se","wls")
  partialName<-"_Raw4_send"
  Translation<-fread("Predata/MatrizTraduccion.csv")
  i<-1
  j<-1
  
  ###INTERRUPTORES
  
  GUARDADO<-1
  Guardado_Especial<-0
  
  
  
  
  print("Inicio Procedimiento")
  ########################### LECTURA ##################################
  
  for(i in 1:length(file)){
    
    print(c(i,file[i]))
    
    
    ################# LECTURA ##################
    #defino el archivo
    Archivo<-c("Predata/",file[i],"/",insideFile[i],partialName,".csv")
    Fpath<-paste(Archivo, sep = '', collapse = '')
    
    #indico el que se esta leyendo
    print("Incio Lectura Archivo")
    print(Fpath)
    
    #Registro el pais y leo el archivo con la excepcion española 
    
    if(file[i]=="Spain"){
      Datos_ini<-read.csv2(Fpath,sep=",")
      Datos_ini$Country<-file[i]
      dbase<-Datos_ini  
      
    }else{
      Datos_ini<-read.csv2(Fpath,sep="\t")
      Datos_ini$Country<-file[i]
      dbase<-Datos_ini}
    
    ############### ARREGLO ESCOCIA ##################
    if(file[i]=="Scotland"){
      names(dbase)[names(dbase) == 'SupQ_11'] <- 'SupQ_11.x'
      names(dbase)[names(dbase) == 'SupQ_12'] <- 'SupQ_11'
      names(dbase)[names(dbase) == 'SupQ_11.x'] <- 'SupQ_12'
      print("APAÑAO")
    }
    
    
    
    #proceso el archivo`
    print("Comienza procesado")
    
    
    ############# INDICE DE INDIFERENCIA #################
    if(file[i]=="Spain"){
      dbase$Q1 <- as.numeric(dbase$SupQ_1 == 17)
      dbase$Q2 <- as.numeric(dbase$SupQ_5 == 76)
      dbase$Q3 <- as.numeric(dbase$SupQ_11 == 236|dbase$SupQ_11==235)
      dbase$Q4 <- as.numeric(dbase$SupQ_4 == 43)
      
    }else if(file[i]=="Lithuania"){
      
      dbase$Q1 <- as.numeric(dbase$SupQ_1 == 9)
      dbase$Q2 <- as.numeric(dbase$SupQ_5 == 52)
      dbase$Q3 <- as.numeric(dbase$SupQ_11 == 251|dbase$SupQ_11==250)
      dbase$Q4 <- as.numeric(dbase$SupQ_4 == 30)
      
    }else if(file[i]=="Scotland"){
      
      dbase$Q1 <- as.numeric(dbase$SupQ_1 == 10)
      dbase$Q2 <- as.numeric(dbase$SupQ_5 == 55)
      dbase$Q3 <- as.numeric(dbase$SupQ_12 == 209 | dbase$SupQ_12==207)
      dbase$Q4 <- as.numeric(dbase$SupQ_4 == 31)
      
    }else{  #para todo el mundo
      dbase$Q1 <- as.numeric(dbase$SupQ_1 == Translation[i,SupQ_1])
      dbase$Q2 <- as.numeric(dbase$SupQ_5 == Translation[i,SupQ_5])
      dbase$Q3 <- as.numeric(dbase$SupQ_11 == Translation[i,SupQ_11.1] | dbase$SupQ_11==Translation[i,SupQ_11.2])
      dbase$Q4 <- as.numeric(dbase$SupQ_4 == Translation[i,SupQ_4])
    }
    
    
    dbase$Close.Party<-NA
    dbase$Close.Party<-dbase$SupQ_1-(min(as.numeric(dbase$SupQ_1),na.rm=T)-1)
    
    dbase$European.Prefered.Party<-NA
    dbase$European.Prefered.Party<-dbase$SupQ_2-(min(as.numeric(dbase$SupQ_2),na.rm=T)-1)
    
    dbase$Reasons<-NA
    dbase$Reasons<-dbase$SupQ_3-(min(as.numeric(dbase$SupQ_3),na.rm=T)-1)
    
    dbase$Last.Voted<-NA
    dbase$Last.Voted<-dbase$SupQ_4-(min(as.numeric(dbase$SupQ_4),na.rm=T)-1)
  
    dbase$National.Preferred<-NA
    dbase$National.Preferred<-dbase$SupQ_5-(min(as.numeric(dbase$SupQ_5),na.rm=T)-1)
    
    dbase$Interest.Politics<-NA
    dbase$Interest.Politics<-dbase$SupQ_11-(min(as.numeric(dbase$SupQ_11),na.rm=T)-1)
    
  
    
    
    
    # 6 y 7 dan igual
    
    
    dbase$Country<-dbase$Country #Arrastro el pais 
  
    
    print("Comienza Partidos")
    
    ############### PARTIDOS ###############
    
    
    hasPrefix<-grepl("^DD_",names(dbase))
    NumPartidos<-sum(hasPrefix, na.rm=TRUE)
    
    #backup
    Backup<-dbase[, hasPrefix]
    
    #actualizo los la probabilidad de votar a algun partido
    #LINEA DE BACK UP NO TOCAR ORIGINALMENTE FUNCIONA CON UN NIVEL() dbase[, hasPrefix]<-lapply(dbase[ ,hasPrefix, drop=FALSE], function(x) ifelse(x >= 10 , 1 , 0))
    #PASAMOS LA CODIFICACION DEL PASITIDO MAXIMO SOBRE EL BACKUP Y 
    #REALIZAMOS LA SUMA SOBRE UNDERIVADO PARA HALLAR EL NUMERO DE PARTIDOS MAXIMOS EN CADA NIVEL
    
    Backup_5<-as.data.frame(lapply(Backup, function(x) ifelse(x >= 5 , 1 , 0)))
    Backup_6<-as.data.frame(lapply(Backup, function(x) ifelse(x >= 6 , 1 , 0)))
    Backup_7<-as.data.frame(lapply(Backup, function(x) ifelse(x >= 7 , 1 , 0)))
    Backup_8<-as.data.frame(lapply(Backup, function(x) ifelse(x >= 8 , 1 , 0)))
    Backup_9<-as.data.frame(lapply(Backup, function(x) ifelse(x >= 9 , 1 , 0)))
    Backup_10<-as.data.frame(lapply(Backup, function(x) ifelse(x >= 10 , 1 , 0)))
    
    
    
    #y agrego
    #LINEA OTRIGINAL NO TOCAR  dbase$TotInt<-rowSums(dbase[,hasPrefix])
    dbase$TotInt_5<-rowSums(Backup_5)
    dbase$TotInt_6<-rowSums(Backup_6)
    dbase$TotInt_7<-rowSums(Backup_7)
    dbase$TotInt_8<-rowSums(Backup_8)
    dbase$TotInt_9<-rowSums(Backup_9)
    dbase$TotInt_10<-rowSums(Backup_10)
    #el relativo de partidos
    dbase$PorcentajePartidos.5<-dbase$TotInt_5/NumPartidos
    dbase$PorcentajePartidos.6<-dbase$TotInt_6/NumPartidos
    dbase$PorcentajePartidos.7<-dbase$TotInt_7/NumPartidos
    dbase$PorcentajePartidos.8<-dbase$TotInt_8/NumPartidos
    dbase$PorcentajePartidos.9<-dbase$TotInt_9/NumPartidos
    dbase$PorcentajePartidos.10<-dbase$TotInt_10/NumPartidos
    #lo que si votarian
    dbase$VoteProb.5<-ifelse(dbase$TotInt_5>=1,1,0)
    dbase$VoteProb.6<-ifelse(dbase$TotInt_6>=1,1,0)
    dbase$VoteProb.7<-ifelse(dbase$TotInt_7>=1,1,0)
    dbase$VoteProb.8<-ifelse(dbase$TotInt_8>=1,1,0)
    dbase$VoteProb.9<-ifelse(dbase$TotInt_9>=1,1,0)
    dbase$VoteProb.10<-ifelse(dbase$TotInt_10>=1,1,0)
    #reincorporo el backup con el prefijo Sub_
    colnames(Backup) <- paste("Sub", colnames(Backup), sep = "_")
    dbase<-cbind(dbase,Backup)
    
    ####Meto las variables control en intermedio y luego lo pego a la grande (dbase)
    Intermedio<-dbase
    
    print("Comienza Variables control")
    if(file[i]=="Scotland"){
      ########GENDER
      Intermedio$Gender<-NA
      Intermedio$Gender<-Intermedio$SupQ_9-(min(as.numeric(dbase$SupQ_9),na.rm=T)-1)
      Intermedio$Gender<-recode(Intermedio$Gender,"3=NA;1=0;2=1")
      ########Foireigner
      Intermedio$Foreigner<-NA
      Intermedio$Foreigner<-Intermedio$SupQ_7
      Intermedio$Foreigner<-recode(Intermedio$Foreigner,"1=NA")
      Intermedio$Foreigner<- ifelse(Intermedio$Foreigner == Translation[i,Foreign] , 0 , 1)
      ####EDUCATION
      Intermedio$Education<-NA
      Intermedio$Education<-Intermedio$SupQ_11-(min(as.numeric(dbase$SupQ_11),na.rm=T)-1)
      Intermedio$Education<-recode(Intermedio$Education,"1=NA")
      Intermedio$University<-ifelse(Intermedio$Education >= 5 , 1 , 0)
      ####Urban
      Intermedio$Urban<-NA
      Intermedio$Urban<-Intermedio$SupQ_13-(min(as.numeric(dbase$SupQ_13),na.rm=T)-1)
      Intermedio$Urban<-recode(Intermedio$Urban,"5=NA;6=NA")
      Intermedio$Ciudad<-ifelse(Intermedio$Education <= 2 , 1 , 0)
      #####AGE
      Intermedio$Age<-NA
      Intermedio$Age<-Intermedio$SupQ_10
      Intermedio$Age<-recode(Intermedio$Age,"98=NA")
      Intermedio$Age<-2014-Intermedio$Age
      #####Religion
      Intermedio$Religion<-NA
      Intermedio$Religion<-Intermedio$SupQ_15-(min(as.numeric(dbase$SupQ_15),na.rm=T)-1)
      Intermedio$Religion<-recode(Intermedio$Religion,"12=NA")
      #####Employment Unemployment
      Intermedio$Employment<-NA
      Intermedio$Employment<-Intermedio$SupQ_14-(min(as.numeric(dbase$SupQ_14),na.rm=T)-1)
      Intermedio$Employment<-recode(Intermedio$Employment,"8=NA;9=NA")
      
    }else{
      
      ########GENDER
      Intermedio$Gender<-NA
      if(file[i]=="Cyprus"){
        Intermedio$Gender<-Intermedio$SupQ_8-(min(as.numeric(dbase$SupQ_8),na.rm=T))
        Intermedio$Gender<-recode(Intermedio$Gender,"3=NA;1=0;2=1")
      }else if (file[i]=="Austria"){ 
        Intermedio$Gender<-Intermedio$SupQ_8-(min(as.numeric(dbase$SupQ_8),na.rm=T)-1)
        Intermedio$Gender<-recode(Intermedio$Gender,"3=NA;7=NA;5=1;6=2;1=0;2=1")
      }else if (file[i]=="Netherlands"){ 
        Intermedio$Gender<-Intermedio$SupQ_8-(min(as.numeric(dbase$SupQ_8),na.rm=T)-1)
        Intermedio$Gender<-recode(Intermedio$Gender,"5=NA;1=NA;2=NA")
        Intermedio$Gender<-Intermedio$Gender-2
        Intermedio$Gender<-recode(Intermedio$Gender,";1=0;2=1")
      }else if (file[i]=="Hungary"){ 
        Intermedio$Gender<-Intermedio$SupQ_8-(min(as.numeric(dbase$SupQ_8),na.rm=T)-1)
        Intermedio$Gender<-recode(Intermedio$Gender,"4=NA;1=NA")
        Intermedio$Gender<-Intermedio$Gender-1
        Intermedio$Gender<-recode(Intermedio$Gender,";1=0;2=1")
      }else{   
        Intermedio$Gender<-Intermedio$SupQ_8-(min(as.numeric(dbase$SupQ_8),na.rm=T)-1)
        Intermedio$Gender<-recode(Intermedio$Gender,"3=NA;4=NA;5=NA")
        Intermedio$Gender<-recode(Intermedio$Gender,";1=0;2=1")}
      ########Foireigner
      Intermedio$Foreigner<-NA
      Intermedio$Foreigner<-Intermedio$SupQ_6
      Intermedio$Foreigner<-recode(Intermedio$Foreigner,"1=NA")
      Intermedio$Foreigner<- ifelse(Intermedio$SupQ_6 == Translation[i,Foreign] , 0 , 1)
      
      ####EDUCATION
      if(file[i]=="Croatia"){
        Intermedio$Education<-NA
        Intermedio$Education<-Intermedio$SupQ_10-(min(as.numeric(dbase$SupQ_10),na.rm=T)-1)
        Intermedio$Education<-recode(Intermedio$Education,"1=NA;8=7")
        Intermedio$Education<-recode(Intermedio$Education,"8=NA;9=NA;10=NA;11=NA;12=NA")
        Intermedio$Education<-Intermedio$Education-1
      }else if(file[i]=="Netherlands"){
        Intermedio$Education<-NA
        Intermedio$Education<-Intermedio$SupQ_10-(min(as.numeric(dbase$SupQ_10),na.rm=T)-1)
        Intermedio$Education<-recode(Intermedio$Education,"1=NA;2=NA;3=NA")
        Intermedio$Education<-Intermedio$Education-4
      }else if(file[i]=="Austria"){
        Intermedio$Education<-NA
        Intermedio$Education<-Intermedio$SupQ_10-(min(as.numeric(dbase$SupQ_10),na.rm=T)-1)
        Intermedio$Education<-recode(Intermedio$Education,"1=NA;8=NA;9=NA;10=NA;11=NA;12=NA;13=NA;14=NA")
        Intermedio$Education<-Intermedio$Education-1
      }else{
        Intermedio$Education<-NA
        Intermedio$Education<-Intermedio$SupQ_10-(min(as.numeric(dbase$SupQ_10),na.rm=T)-1)
        Intermedio$Education<-recode(Intermedio$Education,"1=NA;8=NA;9=NA;10=NA")
        Intermedio$Education<-Intermedio$Education-1
      }
  
      
      ####Urban
      Intermedio$Urban<-NA
      Intermedio$Urban<-Intermedio$SupQ_12-(min(as.numeric(dbase$SupQ_12),na.rm=T)-1)
      Intermedio$Urban<-recode(Intermedio$Urban,"5=NA;6=NA")
  
      
      #####AGE
      Intermedio$Age<-NA
      Intermedio$Age<-Intermedio$SupQ_9
      Intermedio$Age<-recode(Intermedio$Age,"98=NA")
      Intermedio$Age<-2014-Intermedio$Age
      Intermedio$Joven<-ifelse(Intermedio$Age>18 & Intermedio$Age<30 ,1,0)
      #####Religion
      Intermedio$Religion<-NA
      Intermedio$Religion<-Intermedio$SupQ_14-(min(as.numeric(dbase$SupQ_14),na.rm=T)-1)
      Intermedio$Religion<-recode(Intermedio$Religion,"12=NA")
      #####Employment
      Intermedio$Employment<-NA
      Intermedio$Employment<-Intermedio$SupQ_13-(min(as.numeric(dbase$SupQ_13),na.rm=T)-1)
      Intermedio$Unemployment<-ifelse(Intermedio$Employment == 7 , 1 , 0)
      Intermedio$Employment<-recode(Intermedio$Employment,"8=NA;9=NA;10=NA;11=NA;12=NA")
    }
    
    
    if(file[i]=="Spain"){
      Intermedio$Demonstration<-NA
      Intermedio$Demonstration<-Intermedio$SupQ_16-(min(as.numeric(Intermedio$SupQ_16),na.rm=T)-1)
      
      Intermedio$Contacted<-NA
      Intermedio$Contacted<-Intermedio$SupQ_17-(min(as.numeric(Intermedio$SupQ_17),na.rm=T)-1)
      
      Intermedio$ActiveRole<-NA
      Intermedio$ActiveRole<-Intermedio$SupQ_18-(min(as.numeric(Intermedio$SupQ_18),na.rm=T)-1)
      
      Intermedio$Petition<-NA
      Intermedio$Petition<-Intermedio$SupQ_19-(min(as.numeric(Intermedio$SupQ_19),na.rm=T)-1)
      
      Intermedio$Boycot<-NA
      Intermedio$Boycot<-Intermedio$SupQ_20-(min(as.numeric(Intermedio$SupQ_20),na.rm=T)-1)
    }
    
    dbase<-Intermedio
    
    
    
    
    ################# Partido Maximo ##############
    
    hasPrefix          <-    grepl("^DD_",names(dbase))
    dbase$MaxParty     <- apply(dbase[,hasPrefix],1,max) 
    dbase$MaxParty_NA  <- sum(is.na(dbase$MaxParty))
    dbase$MaxParty_Rep <- rowSums(dbase[,hasPrefix] ==  dbase$MaxParty )
    dbase[,"Preferido_Partido_Pref.F"]<-names(dbase)[hasPrefix][max.col(dbase[hasPrefix], 'first')*NA^(rowSums(dbase[hasPrefix])==0)] 
    dbase$Partido_Pref.Comparado.F<-substring(dbase$Preferido_Partido_Pref.F,4)
    
    
    ############# Partido declarado favorito ####################
    print("Comienza favorito")
    Archivo_Preguntas<-c("Predata/",file[i],"/",insideFile[i],"_Codebook",".csv")
    Archivo_Preguntas<-paste(Archivo_Preguntas, sep = '', collapse = '')
    
    
    if(file[i]=="Croatia"){
      Preguntas_Index<-read.csv2(Archivo_Preguntas,sep="\t")
    }else{Preguntas_Index<-fread(Archivo_Preguntas)}
    
    
    
    if(file[i]=="Croatia"){
      
      First_s<-grep("^2$", Preguntas_Index$supplementary_question_id)
      First_f<-grep("^3$", Preguntas_Index$supplementary_question_id)
      First_s1<-First_s[1]
      First_f1<-First_f[1]-1
      Pregunta_Partidos_filtrada<-Preguntas_Index[First_s1:First_f1,]
      Pregunta_Partidos_filtrada<-Pregunta_Partidos_filtrada[,c("supplementary_answer_id" ,"english")]
      dbase$ANS_Party<-as.numeric(dbase$SupQ_2)
      dbase<-merge(x = dbase, y = Pregunta_Partidos_filtrada, by.x = "ANS_Party", by.y="supplementary_answer_id", all.x = TRUE)
    }else if(file[i]=="Spain"){
      
      First_s<-grep("^2$", Preguntas_Index$supplementary_question_id)
      First_f<-grep("^3$", Preguntas_Index$supplementary_question_id)
      First_s1<-First_s[1]
      First_f1<-First_f[1]-1
      Pregunta_Partidos_filtrada<-Preguntas_Index[First_s1:First_f1,]
      Pregunta_Partidos_filtrada<-Pregunta_Partidos_filtrada[,c("supplementary_answer_id" ,"castellano")]
      names(Pregunta_Partidos_filtrada)[2]<-paste("english")
      dbase$ANS_Party<-as.numeric(dbase$SupQ_2)
      dbase<-merge(x = dbase, y = Pregunta_Partidos_filtrada, by.x = "ANS_Party", by.y="supplementary_answer_id", all.x = TRUE)
    }else{
      First_s<-grep("^2$", Preguntas_Index$supplementary_question_id)
      First_f<-grep("^3$", Preguntas_Index$supplementary_question_id)
      First_s1<-First_s[1]
      First_f1<-First_f[1]-1
      Pregunta_Partidos_filtrada<-Preguntas_Index[First_s1:First_f1,]
      Pregunta_Partidos_filtrada<-Pregunta_Partidos_filtrada[,c("supplementary_answer_id" ,"english")]
      dbase$ANS_Party<-as.numeric(dbase$SupQ_2)
      dbase<-merge(x = dbase, y = Pregunta_Partidos_filtrada, by.x = "ANS_Party", by.y="supplementary_answer_id", all.x = TRUE)
    }
    
    
    
    
    dbase$Partido_Maximo<-dbase$Partido_Pref.Comparado.F
    dbase$Partido_Preferido<-dbase$english
    
    #############LIMPIEZA######################
    print("Comienza limpieza") 
    
    Columnas<-c("Country","Record.ID","Attempt","IP.Address",       
                "Browser", "Version","Platform","Mobile",                   
                "Timestamp","language","http_referer","sq1t","sq2t",                     
                "sl2at","sl2bt","sl2ct","dd1t","Total.Time","maxSuccessiveEqualAnswers","less2",                    
                "less3","noOpinions","optint","X.Placement","Y.Placement","Z.Placement",
                "VoteProb.5","VoteProb.6","VoteProb.7","VoteProb.8","VoteProb.9","VoteProb.10",
                "PorcentajePartidos.5","PorcentajePartidos.6",
                "PorcentajePartidos.7","PorcentajePartidos.8","PorcentajePartidos.9",
                "PorcentajePartidos.10","TotInt_5","TotInt_6","TotInt_7",
                "TotInt_8","TotInt_9","TotInt_10","Close.Party","European.Prefered.Party",
                "Reasons","Last.Voted","National.Preferred","Interest.Politics",
                "Gender","Foreigner","Education","Urban","Age","Religion",
                "Employment", "Partido_Maximo","Partido_Preferido")
    
    
    #lo limpio para aligerar el calculo incluyendo las columnas de partidos y el backup para traceability
    Selection<-dbase[,Columnas]
    Selection<-cbind(Selection,dbase[,hasPrefix])
    Selection<-cbind(Selection,Backup)
    Selection<-cbind(Selection,dbase[,c(66:125)]) ###> REFERENCIA NUMERICA OJO
    Selection<-cbind(Selection,dbase[,c(22:35)])
  
  
    
  
    SelLimpia1<-Selection #legacy
    SelLimpia<-SelLimpia1 #legacy
  
    
  
    ########### EUROPEAN MATRIX  ###########
    
    print("Comienza guardado")
    if(GUARDADO==1){ 
      
      EuroMatrix_keep<-Selection[,Columnas ]
      
      if(i==1){
        
        EuroMatrix <-dbase[,Columnas ]
        
      }else{
        
        EuroMatrix <- rbind(EuroMatrix,EuroMatrix_keep)
        
      }
      
      #guardo la matriz local
      
      Archivo<-c("Predata/","/GUILLE/",file[i],"_CURADO",".csv")
      Fpath<-paste(Archivo, sep = '', collapse = '')
      fwrite(dbase,Fpath)
      print("Archivo local Guardado")
      print(Fpath)
      
      #Guardo el archivo Europeo
      
      Archivo<-c("Predata/","/GUILLE/","EUROPA","_CURADO",".csv")
      Fpath<-paste(Archivo, sep = '', collapse = '')
      fwrite(EuroMatrix,Fpath)
      print("Archivo global Guardado")
      
   }
  }
  
