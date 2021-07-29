#include "ConfigurationFile.h"


#include <map>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include <basics/basics/utilities/UbException.h>


namespace vf::basics
{

void ConfigurationFile::clear()
{
   data.clear();
}
//////////////////////////////////////////////////////////////////////////
bool ConfigurationFile::load(const std::string& file)
{
   std::ifstream inFile(file.c_str());

   if (!inFile.good())
   {
      UB_THROW(UbException(UB_EXARGS, "Cannot read configuration file "+file+"!"));
   }

   while (inFile.good() && ! inFile.eof())
   {
      std::string line;
      getline(inFile, line);

      // filter out comments
      if (!line.empty())
      {
         size_t pos = line.find('#');

         if (pos != std::string::npos)
         {
            line = line.substr(0, pos);
         }
      }

      // split line into key and value
      if (!line.empty())
      {
         size_t pos = line.find('=');

         if (pos != std::string::npos)
         {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));

            if (!key.empty() && !value.empty())
            {
               data[key] = value;
            }
         }
      }
   }

   return true;
}

//////////////////////////////////////////////////////////////////////////
template<>
bool ConfigurationFile::fromString<bool>(const std::string& str) const
{
   return str == "true";
}

//////////////////////////////////////////////////////////////////////////
bool ConfigurationFile::contains(const std::string& key) const
{
   return data.find(key) != data.end();
}
//////////////////////////////////////////////////////////////////////////
std::string ConfigurationFile::getString(const std::string& key) const
{
   std::map<std::string, std::string>::const_iterator iter = data.find(key);

   if (iter != data.end())
   {
      std::string value = iter->second;
      return value;
   }
   else
   {
      UB_THROW(UbException(UB_EXARGS, "The parameter \"" + key + "\" is missing!"));
   }
}
//////////////////////////////////////////////////////////////////////////
std::string ConfigurationFile::trim(const std::string& str)
{
   size_t first = str.find_first_not_of(" \t\n\r");

   if (first != std::string::npos)
   {
      size_t last = str.find_last_not_of(" \t\n\r");

      return str.substr(first, last - first + 1);
   }
   else
   {
      return "";
   }
}
//////////////////////////////////////////////////////////////////////////
void ConfigurationFile::split(std::vector<std::string>& lst, const std::string& input, const std::string& separators, bool remove_empty) const
{
   std::ostringstream word;
   for (size_t n = 0; n < input.size(); ++n)
   {
      if (std::string::npos == separators.find(input[n]))
         word << input[n];
      else
      {
         if (!word.str().empty() || !remove_empty)
            lst.push_back(word.str());
         word.str("");
      }
   }
   if (!word.str().empty() || !remove_empty)
      lst.push_back(word.str());
}
//////////////////////////////////////////////////////////////////////////


}