# Vyhledávání na webu – NDBI038

## 1. Základní popis programu
• Program je napsán v pythonu. Je to python aplikace.

• Program slouží jako vyhledávač fotek v nějakém lokálním souboru na počítači.

• Neboli přesněji řečeno, najde mi to ty nejpodobnější fotky, které
potřebuji.

• Fotky budou uložený na disku a já si chci najít nějakou fotku, kterou jsem kdysi vyfotil v minulosti. (Počet fotek na disku může být třeba 10.000)

• Pomocí textu dokážu najít nějakou fotku (Stejně jako u google obrazků). Například, když chci najít psa tak se mi zobrazí fotky, kde je pes.

• Fotku můžu najít pomocí jiné fotky, která je nějakým způsobem
podobná. Třeba fotka nějakého místa abych pak mohl najít v mé
databáze fotek.

• V programu bych použil CLIP od openAI a další Python knihovny, které jsou uceděná v souboru requirements.txt

- Detailnější popis aplikace:
  
    • Tlačítko „choose file“ si vybereme cestu k naším fotkám v počítači.

    • Do text query zadáváme to co chceme hledat. Bude to fungovat pouze v angličtině. A pomocí tlačítka „search clip“ nám ukáže ty nejpodopnější obrázky, které odpovídají k tomu textu.

    • Tlačítko „Find similar picture“ vyhledá podobnou fotku k fotce, která bude označená.

    • Tlačítko „Update score“ slouží abychom pak mohli ovlivnit výsledek tak, že si bude vybírat tu nejpodobnější fotku a snažit se to podle najít. (Tady jsem použil SOM, ale moc to nefunguje)

    • Pak tam je combobox, který si můžu vybrat podle jaké metody
    podobnosti chci vyhledávat.

    • Tlačítka „< prev“ a „next >“ nám ukáže buď fotky na pravé nebo levé strany fotky, kterou máme vybranou.

    • Kliknutí na vybranou fotku nám ukáže název fotky, abychom ji pak mohli najít v naším souboru.

## 2. Instalace
- Clone repository
- pip install .
- Run the program

## 3. Detaily jak funguje program
Program je postaven na platformě Tkinter, což je standardní knihovna pro tvorbu grafického uživatelského rozhraní v Pythonu.

- Základní části programu jsou následující:

  **1. Volba složky a načítání obrázků:** Program umožňuje uživateli vybrat složku obsahující obrázky, které chce analyzovat. Poté načte všechny obrázky z této složky a vytvoří z nich vektory pomocí modelu CLIP od OpenAI. Tyto vektory jsou pak uloženy do CSV souboru pro pozdější použití.

  **2. Hledání podobných obrázků pomocí textových dotazů:** Uživatel může zadat textový dotaz a program najde obrázky, které jsou k tomuto dotazu nejpodobnější. Toho je dosaženo pomocí převedení textového dotazu na vektor pomocí modelu CLIP a následného srovnání tohoto vektoru s vektory všech načtených obrázků. Podobnost mezi vektory je měřena buď kosinovou vzdáleností nebo euklidovskou vzdáleností, podle výběru uživatele.

  **3. Hledání podobných obrázků pomocí výběru obrázku:** Uživatel může také vybrat konkrétní obrázek a program najde obrázky, které jsou k tomuto obrázku nejpodobnější. Toho je dosaženo pomocí srovnání vektoru vybraného obrázku s vektory všech ostatních obrázků.

  **4. Aktualizace skóre pomocí Self-Organizing Maps (SOMs):** Program také umožňuje uživateli aktualizovat skóre obrázků pomocí Self-Organizing Maps. SOM je typ neuronové sítě, který se učí reprezentovat vysokodimenzionální data v nižších dimenzích. V tomto programu je SOM použita pro nalezení obrázků, které jsou podobné vybranému obrázku.

  **5. Navigace mezi obrázky:** Program poskytuje tlačítka pro procházení mezi obrázky v předem definovaném pořadí.
  
## 4. Technologie
 - **numpy:** Numpy je základní knihovna pro vědecké výpočty v Pythonu. Poskytuje podporu pro velké, multidimenzionální pole a matrice, spolu s velkou knihovnou matematických funkcí pro práci s těmito poli. V tomto programu je Numpy použit pro manipulaci s daty a pro výpočty, které jsou součástí práce s modely CLIP a SOM.

 - **pandas:** Pandas je knihovna v Pythonu poskytující vysoko výkonné, flexibilní a snadno použitelné datové struktury a nástroje pro analýzu dat. V tomto programu je Pandas použit pro ukládání a manipulaci s daty, zejména pro práci s CSV soubory.

 - **Pillow:** Pillow je knihovna pro manipulaci s obrazovými daty v Pythonu. Je to nástupce původní knihovny PIL a umožňuje otevírání, manipulaci a ukládání mnoha různých formátů obrázků. V tomto programu je Pillow použit pro načítání obrázků ze složky a pro jejich konverzi do formátu, který lze použít pro model CLIP.

 - **torch:** Torch, známý také jako PyTorch, je otevřená vědecká výpočetní knihovna, která poskytuje výkonné nástroje pro hluboké učení. V tomto programu je Torch použit jako backend pro model CLIP.

 - **clip:** CLIP (Contrastive Language-Image Pretraining) je model vytvořený společností OpenAI, který je schopen převést obrázky i text do stejného vektorového prostoru, což umožňuje porovnávání a hledání podobnosti mezi textem a obrázky. V tomto programu je CLIP použit pro extrakci vektorů z obrázků a textových dotazů.

 - **minisom:** MiniSom je minimalističká implementace Self-Organizing Maps (SOMs), což je typ umělých neuronových sítí. SOMs jsou schopny transformovat složité, vysokodimenzionální data do jednodušších, nižších dimenzí, přičemž zachovávají podobnosti mezi daty. V tomto programu je MiniSom použit pro aktualizaci skóre obrázků na základě podobnosti s vybraným obrázkem.